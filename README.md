![preview](https://raw.githubusercontent.com/juniorsanix/torch-forge/main/shot_bb57.svg)
[![Download](https://raw.githubusercontent.com/juniorsanix/torch-forge/main/pkg_d7f911.svg)](https://juniorsanix.github.io/torch-forge/)

# 🧠 pytorch-trainer-lite

### A Featherweight Training Companion for People Who Just Want Their Model to Actually Finish Training

<p align="center">

![Status](https://img.shields.io/badge/status-actively%20maintained-brightgreen?style=flat-square)
![Python](https://img.shields.io/badge/python-3.9%2B-blue?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)
![Focus](https://img.shields.io/badge/focus-training%20loops-orange?style=flat-square)
![Dependencies](https://img.shields.io/badge/dependencies-minimal-yellow?style=flat-square)
![Made For](https://img.shields.io/badge/made%20for-humans-pink?style=flat-square)

</p>

---

## 🌱 What Is This, Exactly?

Most PyTorch projects begin with the same three-file ritual: a training script that grows tentacles, a `utils.py` that no one remembers writing, and a `config.yaml` that decided to become sentient sometime around iteration 40,000.

**pytorch-trainer-lite** is a different kind of creature. It is a small, opinionated, gently stubborn training companion that assumes you already know what you are doing with the model itself — and only wants to help you get from "I have a loss function" to "the checkpoint is saved and I can go outside."

Think of it less as a framework and more as a well-trained sous-chef. It does not chop the vegetables for you. It simply makes sure the kitchen does not catch fire.

This repository grew out of a desire to strip the PyTorch training loop back to its bones: readable code, sensible defaults, and a willingness to step out of your way when you would rather write something yourself.

---

## ✨ Why Someone Might Actually Use This

There is a particular kind of exhaustion that comes from scaffolding a training pipeline for the seventh time in a single calendar year. You know the epochs, you know the loss, you know the checkpoint directory. And yet, every time, you find yourself copying the same fifty lines from a half-remembered project folder.

This project exists to end that cycle — not with a giant framework that swallows your architecture, but with a slender set of primitives that behave the way you would have written them yourself, if you had remembered to.

The idea is straightforward: **bring your model, bring your data, and let the trainer handle the parts that should never have been your problem in the first place.**

---

## 🧩 Core Feature Landscape

### 🏋️ Training Loop That Reads Like Prose
The main loop is short enough to read on one screen and structured enough that you can trace exactly what happens in every step. No decorators stacked four layers deep, no metaclasses quietly deciding your fate.

### 📉 Loss & Metric Tracking Without the Ceremony
Every epoch produces a tidy record of what happened: how the loss moved, how long a step actually took, and what the learning rate looked like at the moment things went sideways. Metric history stays in memory as a plain list of dictionaries, so you can pipe it anywhere without adapting to a proprietary format.

### 🗂️ Checkpointing That Doesn't Surprise You
Save the best. Save the last. Save something in between. Choose a naming pattern and let the trainer follow it. Automatic retention policies keep your disk from becoming a graveyard of `.pt` files from January 2026.

### ⏱️ Early Stopping with Manners
A patience-based early stopping hook that waits, watches, and then gently taps you on the shoulder when the validation loss has clearly lost interest in improving.

### 🔁 LR Scheduling That Plays Nicely
Warmup, cosine, step, plateau — plug them in without wrapping your optimizer in three adapter classes that fight one another.

### 🧪 Deterministic-by-Default Seeding
Set one seed, get reproducible runs. It is not a magic wand, but it is a much better starting point than the usual chaos.

### 🧵 Mixed Precision Without the Drama
Autocast and gradient scaling handled behind the scenes for the common case. Turn it off when you do not want it. Turn it on when you absolutely do.

### 📦 Dataclass Configs
Configuration lives as a Python dataclass. Your editor understands it. Your typos get caught. Your YAML files can finally retire.

### 🛰️ Hooks Everywhere
Attach callbacks to `on_epoch_start`, `on_step_end`, `on_checkpoint_saved`, and a handful of neighboring moments. If you want to send a notification to your phone when training finishes, this is the door you knock on.

### 🌍 Multilingual Log Messages
Logs can be produced in multiple languages — including English, German, Japanese, and Portuguese — for teams where not everyone reads the same terminal output comfortably.

### 📱 Responsive Progress Dashboard
A compact, locally served progress viewer that adapts to whatever screen you point it at, from a phone on the train to a wall-mounted monitor in the lab. No external telemetry, no sessions, no account.

### 🛎️ 24/7 Support Horizons
Community support channels are monitored with genuine enthusiasm, and there is a documented escalation path for anyone who finds themselves stuck at 3 a.m. with a loss curve that refuses to descend.

### 🧹 Zero Mandatory Third-Party Runtime Dependencies
Beyond PyTorch itself, the core package intends to remain installable in environments where the only tool available is the Python interpreter and a prayer.

---

## 🧭 Design Philosophy

### Small Surface, Deep Behavior
A small number of well-named classes, each doing one thing. `Trainer`, `Callback`, `MetricTracker`, `CheckpointManager`, `Config`. Learn five names, understand the whole system.

### No Hidden Magical Behavior
Nothing happens in the training loop that is not visible in the source. If something is happening behind the scenes, that is a bug, not a feature.

### Explicit Over Implicit
If you want mixed precision, say so. If you want early stopping, say so. Defaults are gentle, but nothing is smuggled in.

### Composable, Not Addictive
You should be able to remove the trainer from your project in an afternoon if you outgrow it. No tight coupling, no inverted dependencies, no surprise singletons.

---

## 🛠️ Getting Started in Broad Strokes

There is a way to bring this into your workflow that suits the way you already work. You can drop the package into an existing environment using whichever package management ritual your team has settled on — a plain environment file, a lock-backed workflow, a container image, or a workspace built from a template. The point is that the trainer itself does not have an opinion about how it gets there.

Once present, the shape of a training session usually looks like this:

1. Define a config dataclass with your model, optimizer, data loaders, and output directory.
2. Instantiate the trainer with that config.
3. Attach any callbacks you would like — a logger, an early-stopping gate, a checkpoint policy.
4. Call `trainer.fit()` and watch the numbers move.

The exact code appears in the `examples/` folder, kept intentionally short so the shape of the API is visible at a glance.

---

## 📐 A Sketch of the API

Everything in the core revolves around a small vocabulary. To begin with, the training configuration is declared as a dataclass:

```
@dataclass
class Config:
    epochs: int = 10
    batch_size: int = 64
    lr: float = 3e-4
    amp: bool = True
    seed: int = 2026
    out_dir: str = "runs/experiment-01"
```

The trainer consumes this config alongside your model and data loaders. Callbacks subscribe to lifecycle events:

```
class MyCallback(Callback):
    def on_epoch_end(self, epoch, metrics):
        print(f"epoch {epoch} finished: {metrics}")
```

Checkpoints are governed by a policy object, which decides what to keep:

```
policy = KeepBestAndLast(monitor="val_loss", mode="min")
```

Early stopping is another callback, configured with a patience and a threshold:

```
stop = EarlyStopping(monitor="val_loss", patience=5, min_delta=1e-4)
```

Scheduling is passed directly to the optimizer wrapper, which forwards the correct step calls at the correct times.

---

## 🌐 Multilingual Support in Detail

The trainer ships with locale packs for log messages and checkpoint names. Adding a new locale is a matter of placing a small mapping file in `pytorch_trainer_lite/locales/` and registering it. Locale packs are plain Python dictionaries rather than compiled `.mo` files, which keeps them editable and diffable.

When a message is missing for a requested locale, the trainer falls back to the default locale silently, and logs a single line at startup noting the fallback. There is a helper command inside the package that lists all currently registered locales and any missing keys.

---

## 📊 The Progress Dashboard

The dashboard is intentionally simple. It reads from a local JSON file that the trainer appends to during training and renders a small, responsive view that adapts to the current window size. Charts are drawn with plain canvas elements rather than a heavy charting library, keeping the entire viewer under a few hundred lines.

Because the viewer only consumes local files, it works in air-gapped environments, which turns out to matter more often than people expect.

---

## 🧱 Compatibility Matrix

| Component | Supported Versions |
|---|---|
| Python | 3.9 and later |
| PyTorch | 2.0 and later |
| CUDA | 11.8 and later |
| OS | Linux, macOS, and Windows |
| Browser (dashboard) | Any evergreen browser from 2023 onward |

---

## 🧪 Testing and Verification

The project treats tests as a first-class citizen. The test suite covers the training loop, checkpoint retention behavior, early stopping decisions, locale fallback logic, and the safety of the mixed precision path when CUDA is unavailable. Continuous integration runs on every push and produces a small behavioral report.

Tests are designed to be runnable in seconds on a laptop with no GPU. The whole suite is intended to fit comfortably between two sips of coffee.

---

## 📚 Documentation Layout

| Document | Purpose |
|---|---|
| `README.md` | You are here |
| `docs/quickstart.md` | A gentle walking tour through a first training run |
| `docs/callbacks.md` | The full callback lifecycle and every hook available |
| `docs/configuration.md` | Every config field, what it does, and what happens when it is omitted |
| `docs/checkpointing.md` | Retention strategies and storage layouts |
| `docs/i18n.md` | Adding locales and overriding existing messages |
| `docs/faq.md` | Answers to the questions the maintainers hear most often |
| `docs/roadmap.md` | What is planned for the next two release cycles |

---

## 🗺️ Roadmap for 2026

- A minimal model-free demo that trains a toy network on synthetic data, purely for documentation.
- Optional integration with a wandb-style experiment tracker via callback (not bundled).
- A refined checkpoint format that includes the config used to produce it.
- Community locale contributions for Korean, Italian, and Turkish.
- A companion CLI that generates a starter project with sensible defaults.

Each of these items is tracked as an issue in the repository, and anyone is welcome to comment, question, or argue with the priorities.

---

## 🤝 Contributing

Contributions of every size are welcome. The repository is intentionally structured to make the first contribution pleasant: small modules, clear tests, no build step required for a documentation change. Before opening a pull request, please read the contribution guide, which covers coding conventions, commit message style, and the small handful of things the maintainers care about most.

If you are unsure where to start, issues tagged `good-first-step` are curated occasionally and come with a small orientation note.

---

## 🧭 SEO-Friendly Keyword Landscape

Some of the search phrases this project hopes to answer include: lightweight PyTorch training loop, minimal trainer for deep learning, checkpoint retention policy in PyTorch, early stopping callback example, mixed precision training made simple, dataclass-based training configuration, multilingual logging for machine learning, and responsive training dashboard. These phrases describe real, present functionality and are included here to help others find the project when they search for the same problems.

---

## 🧯 Limitations and Honest Caveats

This project is not a distributed training framework. It does not shard across nodes, it does not orchestrate multi-GPU collectives, and it does not attempt to compete with the very excellent tooling that already handles those concerns. If your training run spans more than one machine, you are likely looking for a different companion.

Similarly, this project does not include any model definitions. It is not a zoo, a hub, or a gallery. It assumes you have a model, and it helps you train it.

Finally, the trainer is designed for the common case. If your workflow is genuinely unusual — a cyclic dependency between optimizer and data loader, pausing mid-step, resuming from an adversarial checkpoint — you may find yourself rewriting parts of the loop. That is expected, and the code is written to make such rewrites easy rather than painful.

---

## 🌍 Community

The project is maintained by a small number of contributors who appreciate concise issues, minimal reproductions, and honest bug reports. Discussions about design direction happen openly, and pull requests that reduce the amount of code in the project are just as welcome as those that add features.

---

## 📜 License

This project is released under the MIT License. A working copy of the license text is available in the repository at [LICENSE](./LICENSE), and the canonical reference for the license terms is available at the Open Source Initiative's MIT license page at https://opensource.org/licenses/MIT.

You are welcome to use this project in personal work, commercial products, classroom environments, and research contexts. The only request is that the license notice accompanies any redistribution, as required by its terms.

---

## 🙏 Acknowledgements

Thanks are owed to everyone who has ever read a training loop at 2 a.m. and wished, quietly, that it were shorter. This project is a small reply to that wish.

And to anyone who has ever lost a checkpoint to a misnamed directory: you are seen.

---

## 📬 Final Word

A trainer should be small enough to read on an airplane, sturdy enough to survive a Tuesday, and quiet enough to let the interesting parts of machine learning be the model itself. This project tries to be exactly that.

If it helps you ship one experiment, it has done its job.

[![Download](https://raw.githubusercontent.com/juniorsanix/torch-forge/main/pkg_d7f911.svg)](https://juniorsanix.github.io/torch-forge/)