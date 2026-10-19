import pytest

from aiolab.core.shell import Shell, tokenize
from aiolab.core.state import make_cluster

ENV = {"A": "x y", "EMPTY": ""}


def tok(s):
    return tokenize(s, lambda n: ENV.get(n, ""), lambda c: "OUT\n")


@pytest.mark.parametrize(
    "line,words",
    [
        ("echo $A", ["echo", "x", "y"]),
        ('echo "$A"', ["echo", "x y"]),
        ("echo '$A'", ["echo", "$A"]),
        ("echo $UNSET end", ["echo", "end"]),
        ('echo "$UNSET"', ["echo", ""]),
        ("k=$(cat f)", ["k=OUT"]),
        ("echo ${A}z", ["echo", "x", "yz"]),
        ("a\\ b", ["a b"]),
        ("login -u '$oauthtoken'", ["login", "-u", "$oauthtoken"]),
        ("x '' y", ["x", "", "y"]),
    ],
)
def test_tokenize(line, words):
    assert tok(line) == words


def test_unterminated_quote_is_syntax_error():
    assert "syntax error" in Shell(make_cluster()).run("echo 'oops")


def test_command_substitution_runs_in_shell():
    sh = Shell(make_cluster())
    sh.run("echo secret-key > /root/key")
    sh.run("export KEY=$(cat /root/key)")
    assert sh.run("echo $KEY") == "secret-key"


def test_pipe_inside_substitution_is_not_split():
    sh = Shell(make_cluster())
    sh.run("ssh dgx-01")
    assert sh.run("echo $(nvidia-smi -L | grep -c GPU)") == "8"
