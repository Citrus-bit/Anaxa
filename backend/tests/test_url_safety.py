import ipaddress

from deerflow.community.url_safety import is_blocked_address, validate_public_http_url


def test_shared_address_space_is_blocked() -> None:
    assert is_blocked_address(ipaddress.ip_address("100.64.0.1")) is True
    assert validate_public_http_url("http://100.64.0.1/internal") is not None


def test_globally_routable_unicast_address_is_allowed() -> None:
    assert is_blocked_address(ipaddress.ip_address("8.8.8.8")) is False
    assert validate_public_http_url("https://8.8.8.8/") is None


def test_hostname_resolving_to_shared_address_space_is_blocked() -> None:
    result = validate_public_http_url(
        "https://service.example/",
        resolver=lambda _hostname: [ipaddress.ip_address("100.64.0.1")],
    )

    assert result is not None
