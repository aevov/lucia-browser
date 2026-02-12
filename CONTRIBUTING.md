# Contributing to Lucia Browser

Thank you for your interest in contributing to **Lucia**, the open-source community edition of the Nara Web 4.0 browser!

## 🤝 Code of Conduct
We are committed to providing a friendly, safe, and welcoming environment for all. Please be respectful and considerate in all interactions.

## 🛠️ Getting Started

1.  **Fork the repository** on GitHub.
2.  **Clone your fork** locally:
    ```bash
    git clone https://github.com/YOUR_USERNAME/lucia-browser.git
    cd lucia-browser
    ```
3.  **Install dependencies**:
    - Rust (stable)
    - Node.js (LTS)
    - Tauri CLI (`cargo install tauri-cli`)
4.  **Create a branch** for your feature or fix:
    ```bash
    git checkout -b feature/amazing-new-capability
    ```

## 🏗️ Architecture Awareness
Lucia uses a **Hybrid Licensing Model**. Please be aware of where your code lives:
- **UI / Frontend (`src/`)**: MIT License. logic here should be generic and UI-focused.
- **Engine / Core (`crates/`)**: GPL-3.0 License. Logic here (Nara, ACL, QuantumFS) defines the behavior of the orchestration backplane.

##  pull Request Process
1.  Ensure your code builds and tests pass: `cargo test` and `npm test`.
2.  Update documentation if you change behavior.
3.  Open a Pull Request against the `main` branch.
4.  A maintainer from **WPWakanda LLC** will review your code.

## 🐛 Reporting Issues
Please use the GitHub Issue Tracker to report bugs or request features. Be as specific as possible!

---
*Welcome to the Mesh.* 🌐
