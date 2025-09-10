# **Contributing to the Project**

## 👨‍💻 Main Contributor
Dr. Prof. Sergii Kavun is the main contributor to this project, but we welcome contributions from the community!

## Sign our Contributor License Agreement
Contributions to this project must be accompanied by a [Contributor License Agreement, CLA](https://github.com/s-kav/ds_tools/blob/main/CONTRIBUTOR_LICENSE_AGREEMENT.md). You (or your employer) retain the copyright to your contribution; this simply gives us permission to use and redistribute your contributions as part of the project.

If you or your current employer have already signed the Google CLA (even if it was for a different project), you probably don't need to do it again.

## 🤝 How to Contribute
We encourage community contributions! Whether you want to fix bugs, add features, or improve documentation - your help is appreciated.

Before you start:

📋 For new features or major changes, please open an issue first to discuss your ideas

🐛 Bug fixes can be submitted directly as pull requests

# 📝 Contribution Process

## Step 1: Setup
```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/s-kav/ds_tools.git
cd ds_tools
```

## Step 2: Create Feature Branch

```bash
git checkout -b feature/AmazingFeature
```

💡 Use descriptive branch names like feature/add-authentication or bugfix/fix-memory-leak

## Step 3: Make Changes

✅ Write your code

📚 Add appropriate docstrings and comments

🧪 Add tests for new features


## Step 4: Quality Checks

```bash
# Run tests
pytest

# Format code
black .
ruff --fix .
```

## Step 5: Commit & Push

```bash
# Commit with clear, descriptive messages
git commit -m 'Add some AmazingFeature'

# Push to your fork
git push origin feature/AmazingFeature
```

## Step 6: Submit Pull Request

🔄 Open a Pull Request from your fork to the main repository

📄 Provide clear description of changes

🔗 Reference any related issues


# 📋 Code Standards

| Requirement | Description |
|-------------|-------------|
| **PEP8** | Follow Python style guidelines |
| **Tests** | Include tests for new functionality |
| **Docstrings** | Document all functions and classes |
| **Comments** | Explain complex logic |
| **Formatting** | Use `black` and `ruff` for code formatting |


# 🚀 Quick Start Commands

```bash
# Complete workflow example
git checkout -b feature/my-awesome-feature
# ... make your changes ...
pytest                    # Run tests
black .                   # Format code
ruff --fix .             # Fix linting issues
git commit -m "Add my awesome feature"
git push origin feature/my-awesome-feature
# Then open PR on GitHub
```

Thank you for contributing to make this project better! 🙏
