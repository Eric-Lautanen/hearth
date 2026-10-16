//! The verifier: the whole trust decision, in one named type.
//!
//! Everything this crate knows how to check exists elsewhere — [`crate::chain`] builds and
//! validates the path, [`crate::crl`] reads the revocation lists, [`crate::leaf`] checks what
//! the leaf is *for*, [`crate::verify`] does the arithmetic. What was missing was the type a
//! caller can hand to [`crate::ClientHandshake`]: without one, the only verifier in the crate
//! was [`crate::RejectAll`], which can only refuse, so no `https://` path could be opened on
//! this crate's account.
//!
//! # What it decides, in the order it decides it
//!
//! 1. **The chain**, from the wire bytes: framing, each certificate, the structural rules and
//!    the policy, every link's signature, the anchor, every validity window, and revocation.
//! 2. **The leaf**: that it may be a server certificate at all, and that it names the server
//!    the ClientHello asked for — the name comes from the handshake, not from anything the
//!    server sent.
//! 3. **The `CertificateVerify` signature**, over the bytes RFC 8446 §4.4.3 says are signed,
//!    with the leaf's own key.
//!
//! The order is not arbitrary. The chain comes first because the other two are claims about a
//! certificate that has to be trusted before they mean anything: a name match against an
//! unanchored certificate is a match against a certificate anyone could have made. Then the
//! purpose before the name, which is [`crate::leaf`]'s own argument — it is the cheaper test
//! and the more fundamental thing to be told. The signature is last because it is the only
//! one that costs real arithmetic, and because it is the one check that proves the peer holds
//! the key the certificate names rather than merely pointing at it.
//!
//! # What is deliberately *not* here
//!
//! **No anchors, and no root store.** [`PinnedVerifier::new`] takes the anchors as an
//! argument, exactly as [`crate::chain::verify_chain`] does; where they come from — a pinned
//! set shipped with the downloader, or a platform trust reader — is the caller's decision and
//! the caller's review. A verifier that shipped its own roots would be making a trust
//! decision nobody signed off on, and would make the downloader's update path invisible.
//!
//! **No clock.** The instant checked against comes from
//! [`CertificateVerifyInput::validation_time`], which the caller set when the connection was
//! opened, so a test can ask about 2031 and get the same answer forever.
//!
//! **No fetching.** A CRL is bytes the caller supplies; keeping one fresh is a job for
//! something that knows how to talk to the network, and this crate does not.
//!
//! # What it does not claim
//!
//! A success here means: the chain is anchored in a key the caller pinned, every rule this
//! crate implements held, the leaf is for this name and may be a server, and the peer proved
//! it holds the leaf's private key. It does *not* mean the caller pinned the right anchors, or
//! that the CRLs were current in the sense of being fetched recently — only that any list
//! relied on was in its own update window at the instant being checked.

use crate::chain::{self, ChainError, Policy, Revocation};
use crate::client::{CertificateVerifier, CertificateVerifyInput};
use crate::leaf::{self, LeafError};
use crate::verify::{self, VerifyError};
use crate::x509::Certificate;

/// Which stage of the decision refused.
///
/// One variant per stage of [`PinnedVerifier::verify`], carrying the error that stage already
/// produces rather than a string, so a caller can act on the difference between "the path is
/// not anchored" and "the leaf is not for this name" — and so a test can prove *which* check
/// refused rather than only that something did. The [`CertificateVerifier`] implementation
/// renders these into the `String` the trait returns.
///
/// The comparisons are derived because every error it wraps has them
/// ([`crate::chain::ChainError`], [`crate::leaf::LeafError`], [`crate::verify::VerifyError`]),
/// and a caller deciding what to *do* about a refusal wants to match on the variant, not on
/// the message it happens to print.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerifyFailure {
    /// The path itself: framing, parsing, the structural rules, a link's signature, the
    /// anchor, a validity window, or revocation.
    Chain(ChainError),
    /// The leaf is not a server certificate, is not one for the name we asked for, or the
    /// handshake carried no name at all.
    Leaf(LeafError),
    /// The `CertificateVerify` signature does not verify with the leaf's key. `scheme` is the
    /// wire value from the message, carried next to the error so a diagnostic says which
    /// scheme met which key.
    Signature {
        /// The `signature_scheme` the server sent.
        scheme: u16,
        /// Why the arithmetic refused it.
        error: VerifyError,
    },
}

impl std::fmt::Display for VerifyFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Chain(why) => write!(f, "the certificate chain was refused: {why}"),
            Self::Leaf(why) => write!(f, "the leaf certificate was refused: {why}"),
            Self::Signature { scheme, error } => write!(
                f,
                "the CertificateVerify signature ({scheme:#06x}) does not verify: {error}"
            ),
        }
    }
}

impl std::error::Error for VerifyFailure {}

impl From<ChainError> for VerifyFailure {
    fn from(error: ChainError) -> Self {
        Self::Chain(error)
    }
}

impl From<LeafError> for VerifyFailure {
    fn from(error: LeafError) -> Self {
        Self::Leaf(error)
    }
}

/// A [`CertificateVerifier`] over the caller's pinned anchors and revocation information.
///
/// `Pinned` names the property that matters: the anchors are *pins* the caller chose, not a
/// store this crate maintains, and the verifier borrows them for as long as the handshake
/// holds it. Nothing here is copied, so the downloader can parse its root store once at
/// startup and hand the same certificates to every connection.
///
/// An empty anchor set is allowed and refuses every chain — as does
/// `Revocation::Required(&[])`. Neither is an error at construction: a verifier that trusts
/// nothing is a working verifier, and refusing at the call site is the loud version of the
/// same behaviour.
pub struct PinnedVerifier<'anchors, 'crls> {
    anchors: &'anchors [Certificate<'anchors>],
    revocation: Revocation<'crls>,
    policy: Policy,
}

impl<'anchors, 'crls> PinnedVerifier<'anchors, 'crls> {
    /// A verifier over `anchors`, checking `revocation`, under `policy`.
    ///
    /// `revocation` is not optional and has no default: `Unchecked` is a sentence the caller
    /// has to write, which is what stops revocation checking from being the thing that gets
    /// forgotten. The policy comes from [`Policy::default`] unless the caller has a reason to
    /// move a limit.
    pub fn new(
        anchors: &'anchors [Certificate<'anchors>],
        revocation: Revocation<'crls>,
        policy: Policy,
    ) -> Self {
        Self {
            anchors,
            revocation,
            policy,
        }
    }

    /// The three checks, in order, with typed errors.
    ///
    /// This is the whole decision; [`CertificateVerifier::verify_certificate`] is this plus
    /// the `String` rendering the trait asks for, so a caller that wants to act on *why* can
    /// call this directly.
    pub fn verify(&self, input: &CertificateVerifyInput<'_>) -> Result<(), VerifyFailure> {
        let chain = chain::parse_certificate_list(input.certificate_list)?;
        chain::verify_chain(
            &chain,
            self.anchors,
            self.revocation,
            input.validation_time,
            &self.policy,
        )?;
        // The list is not empty: `parse_certificate_list` refuses an empty one and
        // `verify_chain` refuses it again. Indexing rather than `first()` keeps the absence
        // out of this function's type — a `None` arm here would be a branch no test could
        // reach, which is a branch nobody would ever notice losing.
        let leaf_certificate = &chain[0];
        // The leaf's own two checks. The name comes from the ClientHello we sent, carried
        // through the handshake rather than read from a certificate: a name checked against
        // the server's own bytes would be the server checking itself.
        leaf::verify_leaf(leaf_certificate, input.server_name)?;
        verify::verify_handshake_signature(
            input.signature_scheme,
            leaf_certificate.public_key(),
            input.signed_data,
            input.signature,
        )
        .map_err(|error| VerifyFailure::Signature {
            scheme: input.signature_scheme,
            error,
        })
    }
}

impl CertificateVerifier for PinnedVerifier<'_, '_> {
    /// [`PinnedVerifier::verify`], rendered as the trait's `String`.
    ///
    /// The rendering is the plain `Display` of [`VerifyFailure`], because a caller reading a
    /// log wants the same sentence a caller matching on the error would get — one rendering
    /// rather than two that can drift.
    fn verify_certificate(&self, input: &CertificateVerifyInput<'_>) -> Result<(), String> {
        self.verify(input).map_err(|failure| failure.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A verifier that pins nothing, over an input with no certificates in it: the one
    /// decision here that a test can reach without fixtures.
    fn refuses_an_empty_list() -> VerifyFailure {
        let verifier = PinnedVerifier::new(&[], Revocation::Unchecked, Policy::default());
        let input = CertificateVerifyInput {
            validation_time: crate::UnixTime::epoch(),
            certificate_list: &[],
            signature_scheme: 0x0804,
            signature: &[],
            signed_data: &[],
            server_name: None,
        };
        let rendered = verifier.verify_certificate(&input).unwrap_err();
        let failure = verifier.verify(&input).unwrap_err();
        // One rendering, not two that can drift: what a log prints is what a match sees.
        assert_eq!(failure.to_string(), rendered);
        failure
    }

    /// The framing rule. An empty `certificate_list` is refused *by the chain stage*, not by
    /// a missing leaf or an absent name — which is what the variant's own name says and what
    /// a caller sorting failures needs to be true.
    #[test]
    fn an_empty_certificate_list_is_the_chain_stage_refusing_it() {
        let failure = refuses_an_empty_list();
        assert!(matches!(
            failure,
            VerifyFailure::Chain(ChainError::EmptyList)
        ));
        assert_eq!(
            failure.to_string(),
            "the certificate chain was refused: the certificate list is empty"
        );
    }
}
