## Installation of TimesFM

The `timesfm` library depends on `wandb >= 0.24.1`. This version of weights & biases utilizes a new core written in **Go** and **Rust**. Pre-compiled wheels are not available for all OS/Architecture/Python combinations on PyPI. In these cases, the system must build the package from the source distribution (sdist).

### Prerequisites for Source Build

To install successfully, the following compilers must be installed and accessible in the shell's `$PATH`:

1. **Go (1.25.6 or higher):** Required for the `wandb-core` binary.
* Update via Homebrew: `brew upgrade go`


2. **Rust (latest stable):** Required for the `gpu_stats` component.
* Install via Rustup: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
* Activate in current shell: `source "$HOME/.cargo/env"`


### Installation Execution

Once the compilers are available, the installation completes by building the native extensions locally.

```bash
# Load Rust environment
source "$HOME/.cargo/env"

# Install TimesFM (triggers local build of wandb)
uv pip install timesfm
```
