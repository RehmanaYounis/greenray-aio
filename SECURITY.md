# Security policy

## Trust model

GreenRay AIO is a single-user, local learning tool.

- **Scenario files are code.** `checks` are Python expressions evaluated with `eval`, and `setup` may contain Python run with `exec`. Only load scenario files from this repository and your own private solutions repository. Never load them from untrusted sources.
- **Commands you type never reach `eval` or `exec`.** They are parsed by the simulated tools. All YAML is read with `yaml.safe_load`.
- The text editors run your own local editor (`$VISUAL`, `$EDITOR`, `nano`, `vim` or `vi`) on a temporary file, with your user's permissions.
- Progress is stored locally in `~/.greenray/progress.json`. Nothing is sent over the network.

## Reporting a vulnerability

Please report suspected vulnerabilities privately through GitHub's **Report a vulnerability** (Security tab) rather than in a public issue.
