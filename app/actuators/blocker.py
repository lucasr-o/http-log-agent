"""IP blocking.

The default mode is `dry_run`: the decision is recorded in the database blocklist
and nothing is executed on the system. That is deliberate. A project that alters
firewall rules on receiving a POST is dangerous to review and dangerous to run,
and the agent's demonstrable value lies in the decision, not in `iptables`.

`enforce` mode exists and works, but requires the operator to state the command
explicitly in `BLOCK_COMMAND`. There is no built-in command: the correct way to
block depends on the infrastructure, and guessing would be worse than doing
nothing.
"""

from __future__ import annotations

import ipaddress
import logging
import shlex
import subprocess
from dataclasses import dataclass

logger = logging.getLogger(__name__)

COMMAND_TIMEOUT = 10


@dataclass(slots=True)
class BlockResult:
    executed: bool
    mode: str
    detail: str
    # Target rejected during validation. It exists as a field because callers need
    # to tell "did not block because the IP was invalid" apart from "did not block
    # because it is in dry-run", and deciding that by reading the prefix of
    # `detail` coupled control flow to message text.
    invalid_target: bool = False


class IPBlocker:
    def __init__(self, mode: str = "dry_run", command: str = "") -> None:
        self.mode = mode if mode in {"dry_run", "enforce"} else "dry_run"
        self.command = command

    @staticmethod
    def validate_ip(ip: str) -> str:
        """Reject anything that is not a literal IP address.

        The block target comes from an LLM decision, which in turn read data
        controlled by third parties. Validating here is what keeps a log payload
        from becoming command injection in enforce mode.
        """
        return str(ipaddress.ip_address(ip.strip()))

    def block(self, ip: str, reason: str) -> BlockResult:
        try:
            safe_ip = self.validate_ip(ip)
        except ValueError:
            return BlockResult(
                False, self.mode, f"invalid IP address: {ip!r}", invalid_target=True
            )

        if self.mode == "dry_run":
            logger.info("[dry-run] block of %s recorded: %s", safe_ip, reason)
            return BlockResult(
                executed=False,
                mode="dry_run",
                detail="recorded in the blocklist; no system change",
            )

        if not self.command:
            return BlockResult(
                False, "enforce", "BLOCK_COMMAND not configured; block not executed"
            )

        argv = [part.replace("{ip}", safe_ip) for part in shlex.split(self.command)]
        try:
            completed = subprocess.run(
                argv, capture_output=True, text=True, timeout=COMMAND_TIMEOUT, check=False
            )
        except (OSError, subprocess.SubprocessError) as exc:
            logger.exception("failed to execute block of %s", safe_ip)
            return BlockResult(False, "enforce", f"execution failed: {exc}")

        if completed.returncode != 0:
            return BlockResult(
                False, "enforce", f"command returned {completed.returncode}: {completed.stderr[:200]}"
            )
        return BlockResult(True, "enforce", f"command executed: {' '.join(argv)}")
