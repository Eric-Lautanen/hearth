//! Pure-std HTTP/1.1 client for the model downloader.
//!
//! Zero dependencies, zero subprocesses: request building, response parsing
//! (`Content-Length`, `chunked` and close-delimited bodies), redirect
//! following and `Range` resume live here. The byte transport is abstracted
//! behind [`Transport`] so a TLS stream can slot in later without touching
//! any of this logic — plain TCP is the only transport today.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

/// Byte transport: anything readable and writable. `TcpStream` qualifies
/// through the blanket impl below; a future TLS stream just implements
/// `Read` + `Write` and works unchanged.
pub trait Transport: Read + Write {}

impl<T: Read + Write> Transport for T {}

/// Parsed `http(s)://host[:port]/path` (no userinfo, no fragments).
#[derive(Clone, Debug)]
pub struct Url {
    pub scheme: String,
    pub host: String,
    pub port: u16,
    pub path: String,
}

pub fn parse_url(url: &str) -> Result<Url, String> {
    let (scheme, rest) = url
        .split_once("://")
        .ok_or_else(|| "URL must look like https://host/path".to_string())?;
    if scheme != "http" && scheme != "https" {
        return Err(format!("unsupported scheme '{}'", scheme));
    }
    let (authority, path) = match rest.find('/') {
        Some(i) => (&rest[..i], rest[i..].to_string()),
        None => (rest, "/".to_string()),
    };
    if authority.is_empty() {
        return Err("URL has no host".to_string());
    }
    let (host, port) = match authority.rsplit_once(':') {
        Some((h, p)) if !p.is_empty() => {
            let port: u16 = p.parse().map_err(|_| "bad port in URL".to_string())?;
            (h.to_string(), port)
        }
        _ => (
            authority.to_string(),
            if scheme == "https" { 443 } else { 80 },
        ),
    };
    if host.is_empty() {
        return Err("URL has no host".to_string());
    }
    Ok(Url {
        scheme: scheme.to_string(),
        host,
        port,
        path,
    })
}

#[derive(Debug)]
pub struct ResponseHead {
    pub status: u16,
    pub headers: HashMap<String, String>,
}

impl ResponseHead {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers.get(&name.to_lowercase()).map(|s| s.as_str())
    }

    pub fn content_length(&self) -> Option<u64> {
        self.header("content-length")?.trim().parse().ok()
    }

    pub fn is_chunked(&self) -> bool {
        self.header("transfer-encoding")
            .map(|v| v.to_lowercase().contains("chunked"))
            .unwrap_or(false)
    }
}

fn read_head<T: Transport>(s: &mut T, cap: usize) -> Result<(ResponseHead, Vec<u8>), String> {
    let mut buf = Vec::with_capacity(1024);
    let mut one = [0u8; 1];
    loop {
        if buf.len() > cap {
            return Err("response head too large".to_string());
        }
        let n = s.read(&mut one).map_err(|e| format!("head read: {}", e))?;
        if n == 0 {
            return Err("connection closed while reading head".to_string());
        }
        buf.push(one[0]);
        if buf.len() >= 4 && buf[buf.len() - 4..] == *b"\r\n\r\n" {
            break;
        }
    }
    let text = String::from_utf8(buf).map_err(|_| "response head is not UTF-8".to_string())?;
    let mut lines = text.split("\r\n");
    let status_line = lines.next().unwrap_or("");
    let mut parts = status_line.splitn(3, ' ');
    let _version = parts.next().unwrap_or("");
    let status: u16 = parts
        .next()
        .unwrap_or("")
        .parse()
        .map_err(|_| format!("bad status line: {}", status_line))?;
    let mut headers = HashMap::new();
    for line in lines {
        if line.is_empty() {
            continue;
        }
        let (k, v) = line
            .split_once(':')
            .ok_or_else(|| format!("bad header line: {}", line))?;
        headers.insert(k.trim().to_lowercase(), v.trim().to_string());
    }
    Ok((ResponseHead { status, headers }, Vec::new()))
}

fn read_exact_progress<T: Transport>(
    s: &mut T,
    mut remaining: u64,
    cancel: &AtomicBool,
    progress: &mut dyn FnMut(u64),
    sink: &mut dyn Write,
) -> Result<(), String> {
    let mut buf = [0u8; 65536];
    while remaining > 0 {
        if cancel.load(Ordering::Relaxed) {
            return Err("cancelled".to_string());
        }
        let want = remaining.min(buf.len() as u64) as usize;
        let n = s
            .read(&mut buf[..want])
            .map_err(|e| format!("body read: {}", e))?;
        if n == 0 {
            return Err(format!("connection closed with {} bytes left", remaining));
        }
        sink.write_all(&buf[..n])
            .map_err(|e| format!("sink write: {}", e))?;
        remaining -= n as u64;
        progress(n as u64);
    }
    Ok(())
}

fn read_chunked<T: Transport>(
    s: &mut T,
    cancel: &AtomicBool,
    progress: &mut dyn FnMut(u64),
    sink: &mut dyn Write,
) -> Result<(), String> {
    // Chunk framing is line-oriented; read one byte at a time only for the
    // tiny size lines, never for payload data.
    let read_line = |s: &mut T| -> Result<String, String> {
        let mut line = Vec::with_capacity(32);
        let mut one = [0u8; 1];
        loop {
            if line.len() > 1024 {
                return Err("chunk line too long".to_string());
            }
            let n = s.read(&mut one).map_err(|e| format!("chunk head: {}", e))?;
            if n == 0 {
                return Err("connection closed in chunk head".to_string());
            }
            line.push(one[0]);
            if line.len() >= 2 && line[line.len() - 2..] == *b"\r\n" {
                line.pop();
                line.pop();
                break;
            }
        }
        String::from_utf8(line).map_err(|_| "chunk line is not UTF-8".to_string())
    };
    loop {
        let line = read_line(s)?;
        let size_str = line.split(';').next().unwrap_or("").trim();
        let size =
            usize::from_str_radix(size_str, 16).map_err(|_| format!("bad chunk size: {}", line))?;
        if size == 0 {
            // Consume the trailing header block + final CRLF.
            loop {
                let trailer = read_line(s)?;
                if trailer.is_empty() {
                    break;
                }
            }
            return Ok(());
        }
        let mut left = size;
        let mut buf = [0u8; 65536];
        while left > 0 {
            if cancel.load(Ordering::Relaxed) {
                return Err("cancelled".to_string());
            }
            let want = left.min(buf.len());
            let n = s
                .read(&mut buf[..want])
                .map_err(|e| format!("chunk body: {}", e))?;
            if n == 0 {
                return Err("connection closed inside chunk".to_string());
            }
            sink.write_all(&buf[..n])
                .map_err(|e| format!("sink write: {}", e))?;
            left -= n;
            progress(n as u64);
        }
        // Chunk payload is followed by exactly one CRLF.
        let crlf = read_line(s)?;
        if !crlf.is_empty() {
            return Err("missing CRLF after chunk".to_string());
        }
    }
}

fn read_to_end<T: Transport>(
    s: &mut T,
    cancel: &AtomicBool,
    progress: &mut dyn FnMut(u64),
    sink: &mut dyn Write,
) -> Result<(), String> {
    let mut buf = [0u8; 65536];
    loop {
        if cancel.load(Ordering::Relaxed) {
            return Err("cancelled".to_string());
        }
        match s.read(&mut buf) {
            Ok(0) => return Ok(()),
            Ok(n) => {
                sink.write_all(&buf[..n])
                    .map_err(|e| format!("sink write: {}", e))?;
                progress(n as u64);
            }
            Err(e) => return Err(format!("body read: {}", e)),
        }
    }
}

/// GET `url`, appending the body to `sink`.
///
/// * `resume_from`: sends `Range: bytes=N-` and expects 206; a 200 reply
///   means the server ignored the range — nothing is written and
///   `Err("range_ignored")` tells the caller to truncate and retry with 0.
/// * Redirects (301/302/303/307/308 with `Location`) are followed up to 5
///   times, keeping the range request.
/// * `progress(downloaded_total, total_opt)` fires per buffer; `cancel`
///   aborts between buffers. Plain TCP only — an `https://` URL fails
///   loudly until the TLS transport lands.
pub fn get(
    url: &str,
    resume_from: u64,
    timeout: Duration,
    cancel: &AtomicBool,
    progress: &mut dyn FnMut(u64, Option<u64>),
    sink: &mut dyn Write,
) -> Result<(), String> {
    let mut current = url.to_string();
    for _ in 0..6 {
        let u = parse_url(&current)?;
        if u.scheme == "https" {
            return Err("https needs the TLS transport (not landed yet)".to_string());
        }
        let addr = (u.host.as_str(), u.port)
            .to_socket_addrs()
            .map_err(|e| format!("DNS/resolve {}: {}", u.host, e))?
            .next()
            .ok_or_else(|| format!("no address for {}", u.host))?;
        let mut s = TcpStream::connect_timeout(&addr, timeout)
            .map_err(|e| format!("connect {}: {}", addr, e))?;
        s.set_read_timeout(Some(timeout))
            .map_err(|e| format!("set timeout: {}", e))?;
        s.set_write_timeout(Some(timeout))
            .map_err(|e| format!("set timeout: {}", e))?;

        let mut req = format!(
            "GET {} HTTP/1.1\r\nHost: {}\r\nUser-Agent: hearth-ui/1.0\r\nAccept: */*\r\nConnection: close\r\n",
            u.path, u.host
        );
        if resume_from > 0 {
            req.push_str(&format!("Range: bytes={}-\r\n", resume_from));
        }
        req.push_str("\r\n");
        s.write_all(req.as_bytes())
            .map_err(|e| format!("request write: {}", e))?;
        s.flush().map_err(|e| format!("request flush: {}", e))?;

        let (head, _) = read_head(&mut s, 65536)?;
        if matches!(head.status, 301 | 302 | 303 | 307 | 308) {
            let loc = head
                .header("location")
                .ok_or_else(|| "redirect without Location".to_string())?
                .to_string();
            // Relative redirect: resolve against the current URL's origin.
            current = if loc.starts_with("http://") || loc.starts_with("https://") {
                loc
            } else if let Some(stripped) = loc.strip_prefix('/') {
                format!("{}://{}:{}/{}", u.scheme, u.host, u.port, stripped)
            } else {
                // Relative path: merge onto the current directory.
                let base = match u.path.rfind('/') {
                    Some(i) => &u.path[..=i],
                    None => "/",
                };
                format!("{}://{}:{}{}{}", u.scheme, u.host, u.port, base, loc)
            };
            continue;
        }
        if head.status != 200 && head.status != 206 {
            return Err(format!("HTTP {}", head.status));
        }
        if resume_from > 0 && head.status == 200 {
            // Server ignored Range: nothing was written; the caller truncates
            // its partial file and retries with resume_from = 0.
            return Err("range_ignored".to_string());
        }
        let total = head
            .content_length()
            .map(|n| if resume_from > 0 { n + resume_from } else { n });
        let mut wrote = 0u64;
        let mut prog = |n: u64| {
            wrote += n;
            progress(resume_from + wrote, total);
        };
        if head.is_chunked() {
            read_chunked(&mut s, cancel, &mut prog, sink)?;
        } else if let Some(len) = head.content_length() {
            read_exact_progress(&mut s, len, cancel, &mut prog, sink)?;
        } else {
            read_to_end(&mut s, cancel, &mut prog, sink)?;
        }
        return Ok(());
    }
    Err("too many redirects".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as IoWrite;
    use std::net::TcpListener;
    use std::sync::atomic::AtomicBool;

    /// Serve one canned HTTP response, then close. Returns the base URL.
    fn serve_once(response: Vec<u8>) -> (String, std::thread::JoinHandle<()>) {
        let ln = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://127.0.0.1:{}/file", ln.local_addr().unwrap().port());
        let h = std::thread::spawn(move || {
            let (mut sock, _) = ln.accept().unwrap();
            // Slurp the request head so the client never blocks on send.
            let mut buf = [0u8; 4096];
            let mut got = 0;
            while got < buf.len() {
                match sock.read(&mut buf[got..]) {
                    Ok(0) | Err(_) => break,
                    Ok(n) => {
                        got += n;
                        if got >= 4 && buf[..got].windows(4).any(|w| w == b"\r\n\r\n") {
                            break;
                        }
                    }
                }
            }
            sock.write_all(&response).unwrap();
            sock.flush().unwrap();
        });
        (url, h)
    }

    /// Fetch whole body; returns (bytes, last downloaded_total, last total_opt).
    fn fetch(url: &str, resume: u64) -> (Vec<u8>, u64, Option<u64>) {
        let cancel = AtomicBool::new(false);
        let mut out = Vec::new();
        let mut seen = 0u64;
        let mut total = None;
        get(
            url,
            resume,
            Duration::from_secs(10),
            &cancel,
            &mut |got, t| {
                seen = got;
                total = t;
            },
            &mut out,
        )
        .unwrap();
        (out, seen, total)
    }

    #[test]
    fn content_length_body() {
        let body = b"hello-hearth".to_vec();
        let resp = format!(
            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
        )
        .into_bytes()
        .into_iter()
        .chain(body.clone())
        .collect();
        let (url, h) = serve_once(resp);
        let (out, seen, total) = fetch(&url, 0);
        assert_eq!(out, body);
        assert_eq!(seen as usize, body.len());
        assert_eq!(total, Some(body.len() as u64));
        h.join().unwrap();
    }

    #[test]
    fn chunked_body() {
        let resp = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n5\r\nhello\r\n1\r\n \r\n6\r\nhearth\r\n0\r\n\r\n".to_vec();
        let (url, h) = serve_once(resp);
        let (out, seen, total) = fetch(&url, 0);
        assert_eq!(out, b"hello hearth");
        assert_eq!(seen, b"hello hearth".len() as u64);
        assert_eq!(total, None);
        h.join().unwrap();
    }

    #[test]
    fn redirect_chain() {
        let (final_url, h2) = serve_once(
            b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok".to_vec(),
        );
        let path = final_url
            .split_once("/file")
            .unwrap_or(("", ""))
            .0
            .to_string()
            + "/file";
        let resp = format!(
            "HTTP/1.1 302 Found\r\nLocation: {}\r\nConnection: close\r\n\r\n",
            path
        )
        .into_bytes();
        let (url, h1) = serve_once(resp);
        let (out, seen, total) = fetch(&url, 0);
        assert_eq!(out, b"ok");
        assert_eq!(seen, 2);
        assert_eq!(total, Some(2));
        h1.join().unwrap();
        h2.join().unwrap();
    }

    #[test]
    fn range_resume_206() {
        // Client asks from byte 5; server answers 206 with the tail.
        let full = b"0123456789".to_vec();
        let tail = full[5..].to_vec();
        let resp = format!(
            "HTTP/1.1 206 Partial Content\r\nContent-Range: bytes 5-9/10\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            tail.len()
        )
        .into_bytes()
        .into_iter()
        .chain(tail.clone())
        .collect();
        let (url, h) = serve_once(resp);
        let (out, seen, total) = fetch(&url, 5);
        assert_eq!(out, tail);
        // Progress reports resume_base + written; total is the full size.
        assert_eq!(seen, 10);
        assert_eq!(total, Some(10));
        h.join().unwrap();
    }

    #[test]
    fn range_ignored_falls_back() {
        // Server answers 200 to a Range request: caller must restart.
        let body = b"full".to_vec();
        let resp = format!(
            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
        )
        .into_bytes()
        .into_iter()
        .chain(body)
        .collect();
        let (url, h) = serve_once(resp);
        let cancel = AtomicBool::new(false);
        let mut out = Vec::new();
        let err = get(
            &url,
            3,
            Duration::from_secs(10),
            &cancel,
            &mut |_, _| {},
            &mut out,
        )
        .unwrap_err();
        assert_eq!(err, "range_ignored");
        h.join().unwrap();
    }

    #[test]
    fn cancel_aborts() {
        // Server drips body bytes then stalls: the client must abort via the
        // cancel flag instead of hanging on the stalled socket.
        let ln = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://127.0.0.1:{}/file", ln.local_addr().unwrap().port());
        let h = std::thread::spawn(move || {
            let (mut sock, _) = ln.accept().unwrap();
            let mut buf = [0u8; 4096];
            let mut got = 0;
            while got < buf.len() {
                match sock.read(&mut buf[got..]) {
                    Ok(0) | Err(_) => break,
                    Ok(n) => {
                        got += n;
                        if got >= 4 && buf[..got].windows(4).any(|w| w == b"\r\n\r\n") {
                            break;
                        }
                    }
                }
            }
            sock.write_all(
                b"HTTP/1.1 200 OK\r\nContent-Length: 100000000\r\nConnection: close\r\n\r\n",
            )
            .unwrap();
            sock.write_all(&vec![0xABu8; 65536]).unwrap();
            sock.flush().unwrap();
            std::thread::sleep(Duration::from_secs(20));
        });
        let cancel = AtomicBool::new(false);
        let mut out = Vec::new();
        let cancel_ref = &cancel;
        let err = get(
            &url,
            0,
            Duration::from_secs(10),
            cancel_ref,
            &mut |_, _| cancel_ref.store(true, Ordering::Relaxed),
            &mut out,
        )
        .unwrap_err();
        assert_eq!(err, "cancelled");
        assert!(!out.is_empty());
        drop(h);
    }

    #[test]
    fn bad_urls_rejected() {
        assert!(parse_url("ftp://x/y").is_err());
        assert!(parse_url("https://").is_err());
        assert!(parse_url("no-scheme").is_err());
        let u = parse_url("https://huggingface.co/prism-ml/r/resolve/main/f.gguf").unwrap();
        assert_eq!(u.scheme, "https");
        assert_eq!(u.host, "huggingface.co");
        assert_eq!(u.port, 443);
        assert_eq!(u.path, "/prism-ml/r/resolve/main/f.gguf");
    }
}
