# Claude Report - 2026-08-27 - Shared-instance credential exposure

2026-08-27 - Audited who can reach the rented GPU instance and what that exposes.
Workspace: CP (/root/Adhi/BluTrain/dist/Context_Parallelism).

## Finding
/root/.ssh/authorized_keys holds 24 keys. There are NO non-root user accounts
(uid>=1000 list is empty) - everyone logs in as root. Named colleague keys
include madhumithaa.gk, jenifa.d, revathi.t, grishma, nidhin, plus ~19 machine
keys. `last` showed concurrent root sessions from two distinct IPs
(49.207.180.157, 115.246.230.219). sshd: permitrootlogin without-password,
passwordauthentication no.

## Consequence
On a box where others hold root, file permissions are irrelevant - root reads
any file and can dump any process. Therefore:
  * ~/.claude/.credentials.json is a bearer token: anyone with root can run
    Claude AS this account (no password / 2FA / email step) and exhaust quota.
  * GH_TOKEN in ~/.claude-sync.local.conf is readable; if it is a classic PAT
    that is write access to every reachable repo, including BlubridgeAI.
  * Any private SSH key placed there is readable.
  * All chat history under ~/.claude/projects is readable.
There is NO on-box mitigation. Encrypted homes, containers, CLAUDE_CONFIG_DIR
relocation and tmpfs are all defeated by root.

## Mitigations recorded
1. /logout in Claude Code whenever stepping away - shrinks the window to the
   active working period only. Push sessions BEFORE logging out.
2. claude.ai -> log out all devices, to revoke a token already copied.
3. Replace any classic PAT with a fine-grained token scoped to the sessions
   repo only (Contents: read+write); use a per-repo deploy key for BluTrain.
4. Rotate Claude session, PAT and SSH key after any shared box is destroyed.
5. Structural fix: run Claude locally and use the instance for compute over
   ssh/rsync. NOTE: VS Code Remote-SSH does NOT achieve this - it runs the
   extension host (and credentials) on the remote.
6. Real fix for the fleet: per-user accounts (useradd -m, chmod 700 ~) instead
   of 24 people sharing root. Worth escalating: one compromised laptop
   currently compromises every person's credentials.

## Also this session
Launch dir moved to /root/Adhi/BluTrain; sessions re-keyed to
projects/-root-Adhi-BluTrain. Sessions pushed (3c8e650..f4793ca) after --force
takeover from the retired blubridge25-MS-7E06 owner record. Per-box conf was
missing on this instance, which broke logs-push (VAULT_DIR unset) - recreated.
