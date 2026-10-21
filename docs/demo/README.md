# Recording the demo

The README's terminal demo is recorded, not typed by hand, so it can be re-recorded after every release.

Requirements: [asciinema](https://asciinema.org/) to record and [agg](https://github.com/asciinema/agg) to render a GIF. Both run on Linux, WSL and macOS.

```bash
uv tool install asciinema
cargo install --git https://github.com/asciinema/agg   # or download a release binary
```

Record a session in a fresh profile so no progress leaks into the recording, then render:

```bash
./docs/demo/record.sh            # records docs/demo/greenray.cast
agg --cols 100 --rows 28 --font-size 16 --theme monokai docs/demo/greenray.cast docs/demo/greenray.gif
```

`record.sh` opens the `tsh-xid79-drained-node` lab inside asciinema. Type the lines from `session.txt` one at a time, pausing for a second after each output, and finish with `quit`. Keep the whole thing under 60 seconds.
