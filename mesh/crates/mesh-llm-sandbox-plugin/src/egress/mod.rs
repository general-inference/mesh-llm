//! Per-sandbox network egress: the policy a caller sets in the sandbox config,
//! and the host side of the guest's proxy connections ([`proxy`]).
//!
//! Rules, in order:
//!
//! 1. Mode `none` refuses everything.
//! 2. Loopback, link-local (including cloud metadata at 169.254.169.254),
//!    unspecified, multicast and broadcast addresses are never reachable, in
//!    any mode. The host's own services and its mesh ports are on loopback.
//! 3. `deny` entries refuse matching names and addresses.
//! 4. Private and shared ranges (RFC 1918, CGNAT, IPv6 ULA) are reachable only
//!    when an `allow` entry names them, by host or by CIDR, even in `open`.
//! 5. Mode `allowlist` needs an `allow` match; `open` allows the rest.
//!
//! Names are resolved on the host, and the connection goes to an address that
//! passed these checks, so a name cannot be re-pointed after the check.

pub mod proxy;

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};

use anyhow::{Context, Result, bail};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(
    Clone, Copy, Debug, Default, Deserialize, Serialize, JsonSchema, PartialEq, Eq, PartialOrd, Ord,
)]
#[serde(rename_all = "snake_case")]
pub enum EgressMode {
    /// No network access.
    #[default]
    None,
    /// Only destinations matching an `allow` entry.
    Allowlist,
    /// Any public destination not matching a `deny` entry.
    Open,
}

/// The `egress` block of a sandbox config.
#[derive(Clone, Debug, Default, Deserialize, Serialize, JsonSchema, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct EgressSpec {
    #[serde(default)]
    pub mode: EgressMode,
    /// Hosts (`api.example.com`, `*.example.com`, optionally `:port`), IPs,
    /// CIDRs (`10.20.0.0/16`) and presets (`@registries`).
    #[serde(default)]
    pub allow: Vec<String>,
    /// Same forms as `allow`; checked first.
    #[serde(default)]
    pub deny: Vec<String>,
}

/// Package registries and source hosts that installs and clones need.
const REGISTRIES: &[&str] = &[
    "registry.npmjs.org",
    "registry.yarnpkg.com",
    "pypi.org",
    "files.pythonhosted.org",
    "crates.io",
    "index.crates.io",
    "static.crates.io",
    "static.rust-lang.org",
    "proxy.golang.org",
    "sum.golang.org",
    "registry-1.docker.io",
    "auth.docker.io",
    "production.cloudflare.docker.com",
    "ghcr.io",
    "pkg-containers.githubusercontent.com",
    "github.com",
    "codeload.github.com",
    "objects.githubusercontent.com",
    "raw.githubusercontent.com",
    "release-assets.githubusercontent.com",
    "deb.debian.org",
    "security.debian.org",
    "rubygems.org",
    "index.rubygems.org",
    "repo.maven.apache.org",
    "repo1.maven.org",
    "huggingface.co",
    "cdn-lfs.huggingface.co",
];

#[derive(Clone, Debug, PartialEq, Eq)]
enum Target {
    Exact(String),
    /// `*.example.com`: any name under `example.com`, not `example.com` itself.
    Subdomains(String),
    Network {
        addr: IpAddr,
        prefix: u8,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Rule {
    target: Target,
    port: Option<u16>,
}

/// A compiled [`EgressSpec`].
#[derive(Clone, Debug)]
pub struct EgressPolicy {
    mode: EgressMode,
    allow: Vec<Rule>,
    deny: Vec<Rule>,
}

/// Why a connection was refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Denial(pub String);

impl EgressPolicy {
    pub fn compile(spec: &EgressSpec) -> Result<Self> {
        Ok(Self {
            mode: spec.mode,
            allow: parse_entries(&spec.allow).context("egress.allow")?,
            deny: parse_entries(&spec.deny).context("egress.deny")?,
        })
    }

    /// Checks the requested name before resolving it. Returns whether an
    /// `allow` entry named the host, which [`Self::admit`] needs.
    pub fn check_name(&self, host: &str, port: u16) -> Result<bool, Denial> {
        if self.mode == EgressMode::None {
            return Err(Denial("network access is off for this sandbox".into()));
        }
        let host = normalize_host(host);
        if self.deny.iter().any(|rule| rule.matches_name(&host, port)) {
            return Err(Denial(format!("{host}:{port} matches a deny entry")));
        }
        Ok(self.allow.iter().any(|rule| rule.matches_name(&host, port)))
    }

    /// Filters resolved addresses down to the ones this sandbox may reach.
    pub fn admit(
        &self,
        host: &str,
        named_by_allow: bool,
        addrs: &[SocketAddr],
    ) -> Result<Vec<SocketAddr>, Denial> {
        let mut first_denial = None;
        let admitted: Vec<SocketAddr> = addrs
            .iter()
            .copied()
            .filter(|addr| match self.admit_addr(named_by_allow, *addr) {
                Ok(()) => true,
                Err(denial) => {
                    first_denial.get_or_insert(denial);
                    false
                }
            })
            .collect();
        if admitted.is_empty() {
            return Err(first_denial
                .unwrap_or_else(|| Denial(format!("{host} did not resolve to any address"))));
        }
        Ok(admitted)
    }

    fn admit_addr(&self, named_by_allow: bool, addr: SocketAddr) -> Result<(), Denial> {
        let ip = canonical_ip(addr.ip());
        let port = addr.port();
        if is_never_reachable(ip) {
            return Err(Denial(format!(
                "{ip} is a loopback, link-local or reserved address"
            )));
        }
        if self.deny.iter().any(|rule| rule.matches_ip(ip, port)) {
            return Err(Denial(format!("{ip}:{port} matches a deny entry")));
        }
        let allowed_by_ip = self.allow.iter().any(|rule| rule.matches_ip(ip, port));
        let explicitly_allowed = named_by_allow || allowed_by_ip;
        if is_internal(ip) && !explicitly_allowed {
            return Err(Denial(format!(
                "{ip} is a private address; name it in egress.allow to reach it"
            )));
        }
        if self.mode == EgressMode::Allowlist && !explicitly_allowed {
            return Err(Denial(format!("{ip}:{port} is not in egress.allow")));
        }
        Ok(())
    }
}

impl Rule {
    fn matches_name(&self, host: &str, port: u16) -> bool {
        if self.port.is_some_and(|wanted| wanted != port) {
            return false;
        }
        match &self.target {
            Target::Exact(name) => name == host,
            Target::Subdomains(parent) => host
                .strip_suffix(parent.as_str())
                .is_some_and(|prefix| !prefix.is_empty()),
            Target::Network { .. } => host
                .parse::<IpAddr>()
                .is_ok_and(|ip| self.matches_ip(canonical_ip(ip), port)),
        }
    }

    fn matches_ip(&self, ip: IpAddr, port: u16) -> bool {
        if self.port.is_some_and(|wanted| wanted != port) {
            return false;
        }
        match self.target {
            Target::Network { addr, prefix } => in_network(ip, addr, prefix),
            Target::Exact(_) | Target::Subdomains(_) => false,
        }
    }
}

fn parse_entries(entries: &[String]) -> Result<Vec<Rule>> {
    let mut rules = Vec::new();
    for entry in entries {
        let entry = entry.trim();
        if let Some(preset) = entry.strip_prefix('@') {
            rules.extend(preset_rules(preset)?);
            continue;
        }
        rules.push(parse_entry(entry).with_context(|| format!("entry `{entry}`"))?);
    }
    Ok(rules)
}

fn preset_rules(name: &str) -> Result<Vec<Rule>> {
    match name {
        "registries" => Ok(REGISTRIES
            .iter()
            .map(|host| Rule {
                target: Target::Exact((*host).to_string()),
                port: None,
            })
            .collect()),
        other => bail!("unknown egress preset `@{other}`; known presets: @registries"),
    }
}

fn parse_entry(entry: &str) -> Result<Rule> {
    if entry.is_empty() {
        bail!("empty entry");
    }
    if let Some((addr, prefix)) = entry.split_once('/') {
        let addr: IpAddr = addr.parse().context("bad network address")?;
        let prefix: u8 = prefix.parse().context("bad prefix length")?;
        let max = if addr.is_ipv4() { 32 } else { 128 };
        if prefix > max {
            bail!("prefix /{prefix} is longer than {max}");
        }
        return Ok(Rule {
            target: Target::Network {
                addr: canonical_ip(addr),
                prefix,
            },
            port: None,
        });
    }
    if let Ok(ip) = entry.parse::<IpAddr>() {
        return Ok(single_address(ip, None));
    }
    let (host, port) = split_port(entry)?;
    if let Ok(ip) = host.trim_matches(['[', ']']).parse::<IpAddr>() {
        return Ok(single_address(ip, port));
    }
    let host = normalize_host(host);
    if host.contains('*') {
        let Some(parent) = host.strip_prefix("*.") else {
            bail!("wildcards are only allowed as a leading `*.`");
        };
        if parent.is_empty() || parent.contains('*') {
            bail!("wildcards are only allowed as a leading `*.`");
        }
        return Ok(Rule {
            target: Target::Subdomains(format!(".{parent}")),
            port,
        });
    }
    Ok(Rule {
        target: Target::Exact(host),
        port,
    })
}

fn single_address(ip: IpAddr, port: Option<u16>) -> Rule {
    let ip = canonical_ip(ip);
    Rule {
        target: Target::Network {
            addr: ip,
            prefix: if ip.is_ipv4() { 32 } else { 128 },
        },
        port,
    }
}

fn split_port(entry: &str) -> Result<(&str, Option<u16>)> {
    match entry.rsplit_once(':') {
        Some((host, port)) if !host.contains(':') || host.ends_with(']') => {
            Ok((host, Some(port.parse().context("bad port")?)))
        }
        _ => Ok((entry, None)),
    }
}

fn normalize_host(host: &str) -> String {
    host.trim_end_matches('.').to_ascii_lowercase()
}

/// Treats IPv4-mapped IPv6 addresses as the IPv4 address they carry.
fn canonical_ip(ip: IpAddr) -> IpAddr {
    match ip {
        IpAddr::V6(v6) => v6.to_ipv4_mapped().map_or(IpAddr::V6(v6), IpAddr::V4),
        IpAddr::V4(_) => ip,
    }
}

fn in_network(ip: IpAddr, network: IpAddr, prefix: u8) -> bool {
    match (ip, network) {
        (IpAddr::V4(ip), IpAddr::V4(network)) => {
            let mask = u32::MAX.checked_shl(32 - u32::from(prefix)).unwrap_or(0);
            u32::from(ip) & mask == u32::from(network) & mask
        }
        (IpAddr::V6(ip), IpAddr::V6(network)) => {
            let mask = u128::MAX.checked_shl(128 - u32::from(prefix)).unwrap_or(0);
            u128::from(ip) & mask == u128::from(network) & mask
        }
        _ => false,
    }
}

fn is_never_reachable(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => {
            v4.is_loopback()
                || v4.is_link_local()
                || v4.is_unspecified()
                || v4.is_multicast()
                || v4.is_broadcast()
                || in_network(ip, IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0)), 8)
        }
        IpAddr::V6(v6) => {
            v6.is_loopback() || v6.is_unspecified() || v6.is_multicast() || is_v6_link_local(v6)
        }
    }
}

fn is_v6_link_local(v6: Ipv6Addr) -> bool {
    in_network(
        IpAddr::V6(v6),
        IpAddr::V6(Ipv6Addr::new(0xfe80, 0, 0, 0, 0, 0, 0, 0)),
        10,
    )
}

/// Ranges that belong to someone's private network rather than the internet.
fn is_internal(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => {
            v4.is_private() || in_network(ip, IpAddr::V4(Ipv4Addr::new(100, 64, 0, 0)), 10)
        }
        IpAddr::V6(_) => in_network(
            ip,
            IpAddr::V6(Ipv6Addr::new(0xfc00, 0, 0, 0, 0, 0, 0, 0)),
            7,
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy(mode: EgressMode, allow: &[&str], deny: &[&str]) -> EgressPolicy {
        EgressPolicy::compile(&EgressSpec {
            mode,
            allow: allow.iter().map(ToString::to_string).collect(),
            deny: deny.iter().map(ToString::to_string).collect(),
        })
        .unwrap()
    }

    fn addr(text: &str) -> SocketAddr {
        text.parse().unwrap()
    }

    fn reach(policy: &EgressPolicy, host: &str, port: u16, resolved: &str) -> Result<(), Denial> {
        let named = policy.check_name(host, port)?;
        policy.admit(host, named, &[addr(resolved)]).map(|_| ())
    }

    #[test]
    fn mode_none_refuses_everything() {
        let policy = policy(EgressMode::None, &["example.com"], &[]);
        assert!(reach(&policy, "example.com", 443, "93.184.216.34:443").is_err());
    }

    #[test]
    fn allowlist_matches_hosts_ports_wildcards_and_presets() {
        let policy = policy(
            EgressMode::Allowlist,
            &["api.example.com:443", "*.cdn.example.net", "@registries"],
            &[],
        );
        let public = "93.184.216.34:443";
        assert!(reach(&policy, "api.example.com", 443, public).is_ok());
        assert!(reach(&policy, "API.Example.com.", 443, public).is_ok());
        assert!(reach(&policy, "api.example.com", 80, "93.184.216.34:80").is_err());
        assert!(reach(&policy, "a.b.cdn.example.net", 443, public).is_ok());
        assert!(reach(&policy, "cdn.example.net", 443, public).is_err());
        assert!(reach(&policy, "registry.npmjs.org", 443, public).is_ok());
        assert!(reach(&policy, "evil.com", 443, public).is_err());
    }

    #[test]
    fn deny_wins_over_allow_and_open() {
        let policy = policy(
            EgressMode::Open,
            &[],
            &["*.tracking.example.com", "203.0.113.0/24"],
        );
        assert!(reach(&policy, "x.tracking.example.com", 443, "93.184.216.34:443").is_err());
        assert!(reach(&policy, "fine.example.com", 443, "203.0.113.9:443").is_err());
        assert!(reach(&policy, "fine.example.com", 443, "93.184.216.34:443").is_ok());
    }

    #[test]
    fn loopback_link_local_and_metadata_are_never_reachable() {
        let policy = policy(
            EgressMode::Open,
            &["169.254.169.254", "127.0.0.0/8", "localhost"],
            &[],
        );
        for target in [
            "127.0.0.1:3131",
            "169.254.169.254:80",
            "[::1]:443",
            "[fe80::1]:443",
            "0.0.0.0:80",
            "[::ffff:127.0.0.1]:3131",
        ] {
            assert!(
                reach(&policy, "localhost", 80, target).is_err(),
                "{target} should be unreachable"
            );
        }
    }

    #[test]
    fn private_ranges_need_an_explicit_allow_even_when_open() {
        let open = policy(EgressMode::Open, &[], &[]);
        assert!(reach(&open, "nas.lan", 445, "192.168.1.10:445").is_err());
        assert!(reach(&open, "tailnet", 22, "100.100.1.1:22").is_err());
        assert!(reach(&open, "ula", 443, "[fd12::1]:443").is_err());

        let allowed = policy(EgressMode::Open, &["10.20.0.0/16", "git.internal"], &[]);
        assert!(reach(&allowed, "build.internal", 443, "10.20.3.4:443").is_ok());
        assert!(reach(&allowed, "git.internal", 443, "192.168.1.10:443").is_ok());
        assert!(reach(&allowed, "other.internal", 443, "10.21.0.1:443").is_err());
    }

    #[test]
    fn only_admitted_addresses_are_returned() {
        let policy = policy(EgressMode::Open, &[], &[]);
        let admitted = policy
            .admit(
                "mixed.example.com",
                false,
                &[addr("127.0.0.1:443"), addr("93.184.216.34:443")],
            )
            .unwrap();
        assert_eq!(admitted, vec![addr("93.184.216.34:443")]);
    }

    #[test]
    fn bad_entries_and_unknown_presets_are_rejected() {
        for entry in [
            "",
            "a.*.com",
            "*",
            "10.0.0.0/33",
            "host:notaport",
            "@everything",
        ] {
            let spec = EgressSpec {
                mode: EgressMode::Allowlist,
                allow: vec![entry.to_string()],
                deny: Vec::new(),
            };
            assert!(
                EgressPolicy::compile(&spec).is_err(),
                "`{entry}` should be rejected"
            );
        }
    }

    #[test]
    fn modes_order_from_least_to_most_access() {
        assert!(EgressMode::None < EgressMode::Allowlist);
        assert!(EgressMode::Allowlist < EgressMode::Open);
    }
}
