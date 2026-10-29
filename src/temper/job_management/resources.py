"""Load resource profiles and render runtime activation commands.

Schema classes remain re-exported here for existing Python callers.
"""

from pathlib import Path
import shlex
from ruamel.yaml import YAML
from temper.job_management.schemas.resources import (
    RemoteResource,
    RuntimeEnvironment,
)


def load_resource_profile(path: str | Path) -> RemoteResource:
    """Read a user resource profile and resolve local authentication
    paths.

    Parameters
    ----------
    path : str or Path
        JSON or YAML file. Relative paths use the current working
        directory; a leading tilde expands to the submission user's home
        directory.

    Returns
    -------
    RemoteResource
        Validated settings. Relative key/known-hosts paths resolve
        against the profile directory. The original files are
        referenced, never copied.

    Raises
    ------
    ValueError
        If settings are invalid, the document is not a mapping, or it
        specifies a default/internal environment instead of per-family
        environments.
    OSError
        If the profile cannot be read.
    ruamel.yaml.YAMLError
        If the document is not valid JSON or YAML.
    """
    path = Path(path).expanduser().resolve()
    data = YAML(typ="safe").load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(
            "A resource profile must contain a mapping of settings."
        )
    if "environment" in data or "resolved_environment" in data:
        raise ValueError(
            "Resource files must specify environments keyed by MLFF family."
        )
    for key in ("key_filename", "known_hosts"):
        if data.get(key):
            local = Path(data[key]).expanduser()
            data[key] = str((path.parent / local).resolve())
    return RemoteResource.model_validate(data)


def render_environment(environment: RuntimeEnvironment) -> list[str]:
    """Render the activation commands for one resolved Bash environment.

    Parameters
    ----------
    environment : RuntimeEnvironment
        Validated module, Conda/venv, and non-secret variable settings.
        Paths refer to the execution host, not the local profile
        directory.

    Returns
    -------
    list[str]
        Quoted Bash commands in execution order. Run with ``set -e`` so
        missing tools or failed activation stop the task. No commands
        run here and no user shell configuration is modified. When
        cache_root is set, execution creates an owned temporary child
        and installs an EXIT trap: successful tasks remove that child,
        failed tasks retain it and print its path. Do not replace the
        shell with exec after these lines, because doing so bypasses the
        cleanup trap. Existing cache variables are preserved; explicitly
        configured variables are applied first. Optional executable and
        pip checks run after activation.
    """
    lines = [
        f"module load {shlex.quote(module)}" for module in environment.modules
    ]
    if environment.conda_environment:
        if environment.conda_init:
            lines.append(f"source {shlex.quote(environment.conda_init)}")
        else:
            lines += [
                "temper_conda_hook=$(conda shell.bash hook)",
                'eval "$temper_conda_hook"',
                "unset temper_conda_hook",
            ]
        lines.append(
            f"conda activate {shlex.quote(environment.conda_environment)}"
        )
    if environment.venv_path:
        lines.append(
            f"source {shlex.quote(environment.venv_path + '/bin/activate')}"
        )
    lines += [
        f"export {key}={shlex.quote(value)}"
        for key, value in environment.variables.items()
    ]
    lines += [
        f"command -v {shlex.quote(name)} >/dev/null"
        for name in environment.required_executables
    ]
    if environment.check_dependencies:
        lines.append("python -m pip check")
    if environment.cache_root:
        cache_template = environment.cache_root.rstrip("/")
        cache_template += "/temper-cache.XXXXXXXX"
        lines += [
            f"mkdir -p -- {shlex.quote(environment.cache_root)}",
            f"temper_cache=$(mktemp -d {shlex.quote(cache_template)})",
            (
                'trap \'temper_exit=$?; if [ "$temper_exit" -eq 0 ]; then rm'
                ' -rf -- "$temper_cache"; else printf "Retained cache'
                ' diagnostics: %s\\n" "$temper_cache" >&2; fi\' EXIT'
            ),
        ]
        for key, directory in {
            "TRITON_CACHE_DIR": "triton",
            "TORCHINDUCTOR_CACHE_DIR": "inductor",
            "TORCH_EXTENSIONS_DIR": "extensions",
            "CUDA_CACHE_PATH": "cuda",
            "XDG_CACHE_HOME": "xdg",
        }.items():
            lines.append(
                f'export {key}="${{{key}-$temper_cache/{directory}}}"'
            )
    return lines
