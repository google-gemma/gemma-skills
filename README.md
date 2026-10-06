# gemma-skills

Skills for the Gemma and model/agent interactions

## Skills in this repo

| Skill | Description |
| :--- | :--- |
| [`gemma-dev`](skills/gemma-dev) | Skill for building application with Gemma or for general knowledge inquiries related to Gemma models |
| [`gemma-trainer`](skills/gemma-trainer) | Skill for training, fine-tuning, or adapt Gemma models (e.g. SFT, DPO. RLHF, Reward Modeling) on local hardware |

## Installation

You can install these skills across your preferred AI coding assistants and package managers:

### Using [Vercel skills CLI](https://skills.sh)

```sh
# Interactively browse and install skills.
npx skills add google-gemma/gemma-skills --list

# Install a specific skill (e.g., gemma-dev).
npx skills add google-gemma/gemma-skills --skill gemma-dev
```

### Using [Context7 skills CLI](https://context7.com)

```sh
# Interactively browse and install skills.
npx ctx7 skills install /google-gemma/gemma-skills

# Install a specific skill (e.g., gemma-dev).
npx ctx7 skills install /google-gemma/gemma-skills gemma-dev
```

### Antigravity

Install through the AGY CLI:

```sh
# Install it directly
agy plugin install https://github.com/google-gemma/gemma-skills
```

## Disclaimer

This is not an officially supported Google product. This project is not
eligible for the [Google Open Source Software Vulnerability Rewards
Program](https://bughunters.google.com/open-source-security).
