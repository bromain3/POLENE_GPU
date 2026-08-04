import matplotlib.pyplot as plt
import math

class TrainerStats:
    def __init__(self):
        self.loss = []
        self.valLoss = []
        self.lr = []
        self.ranks = []
        self.grads = []
        self.activations = []
        self.perOutputGrads = []


class _statsSave:
    def __init__(self):
        self.stats = TrainerStats()

    def plotRanks(self, use_gram=False, normalized=True, show_baseline=True, overlay_loss=True):
        if not self.stats.ranks:
            return

        epochs = list(range(len(self.stats.ranks)))
        param_names = list(self.stats.ranks[0].keys())

        series = {}
        baselines = {}

        for name in param_names:
            series[name] = [ep[name]["erank"] for ep in self.stats.ranks]
            baselines[name] = self.stats.ranks[0][name]["max_rank"]

        mode_label = "Gram W Wᵀ" if use_gram else "W directe"
        norm_label = " (normalisé)" if normalized else ""
        y_label = f"Rang effectif{norm_label}"

        n_layers = len(series)
        cmap = plt.cm.get_cmap("tab20", n_layers)
        colors = {name: cmap(i) for i, name in enumerate(param_names)}

        # ── figure principale : rang effectif ───────────────────────────────
        if overlay_loss and hasattr(self.stats, "train_loss") and self.stats.train_loss:
            fig, (ax_rank, ax_loss) = plt.subplots(
                2, 1, figsize=(12, 9),
                sharex=True,
                gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08}
            )
        else:
            fig, ax_rank = plt.subplots(figsize=(12, 6))
            ax_loss = None

        for name, values in series.items():
            color = colors[name]
            label = name.replace(".weight", "")

            ax_rank.plot(epochs, values, label=label, color=color, linewidth=1.8)

            # baseline rang max (pointillés légers, même couleur)
            if show_baseline and not normalized:
                ax_rank.axhline(
                    y=baselines[name],
                    color=color,
                    linestyle=":",
                    linewidth=0.8,
                    alpha=0.4,
                )

        # annotation baseline générique
        if show_baseline and not normalized:
            ax_rank.plot([], [], "k:", linewidth=0.8, alpha=0.5, label="rang max (baseline)")

        ax_rank.set_title(
            f"Rang effectif des poids — SVD sur {mode_label}{norm_label}",
            fontsize=13, pad=10
        )
        ax_rank.set_ylabel(y_label)
        ax_rank.grid(True, alpha=0.3)
        ax_rank.legend(fontsize=8, ncol=2, loc="best")

        if normalized:
            ax_rank.set_ylim(0, 1.05)
            ax_rank.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

        # ── overlay loss ─────────────────────────────────────────────────────
        if ax_loss is not None:
            loss_epochs = list(range(len(self.stats.train_loss)))
            ax_loss.plot(loss_epochs, self.stats.train_loss, color="#e05c5c",
                         linewidth=1.8, label="train loss")

            if hasattr(self.stats, "val_loss") and self.stats.val_loss:
                val_epochs = list(range(len(self.stats.val_loss)))
                ax_loss.plot(val_epochs, self.stats.val_loss, color="#e09a5c",
                             linewidth=1.8, linestyle="--", label="val loss")

            ax_loss.set_xlabel("Epoch")
            ax_loss.set_ylabel("Loss")
            ax_loss.grid(True, alpha=0.3)
            ax_loss.legend(fontsize=8, loc="best")
        else:
            ax_rank.set_xlabel("Epoch")

        plt.tight_layout()
        plt.show()

    def plotGrads(
            self,
            vanishing_threshold: float = 1e-4,
            exploding_threshold: float = 10.0,
            overlay_loss: bool = True,
    ):
        """
        Trace la norme L2 moyenne des gradients par couche au fil des époques,
        en échelle log, avec des zones colorées pour les régimes
        vanishing (< vanishing_threshold) et exploding (> exploding_threshold).

        Args:
            vanishing_threshold : norme en-dessous de laquelle le gradient
                                  est considéré comme "vanishing" (défaut 1e-4)
            exploding_threshold : norme au-dessus de laquelle le gradient
                                  est considéré comme "exploding" (défaut 10.0)
            overlay_loss        : affiche la train/val loss en sous-graphe
        """
        if not self.stats.grads:
            print("Aucune donnée de gradient enregistrée.")
            return

        epochs = list(range(len(self.stats.grads)))
        param_names = list(self.stats.grads[0].keys())

        series = {
            name: [ep[name]["grad_norm"] for ep in self.stats.grads]
            for name in param_names
        }

        n_layers = len(series)
        cmap = plt.cm.get_cmap("tab20", n_layers)
        colors = {name: cmap(i) for i, name in enumerate(param_names)}

        if overlay_loss and self.stats.loss:
            fig, (ax, ax_loss) = plt.subplots(
                2, 1, figsize=(12, 9),
                sharex=True,
                gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
            )
        else:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax_loss = None

        # ── zones vanishing / exploding ───────────────────────────────────────
        # On calcule les limites y *après* avoir tracé les courbes pour que
        # axhspan couvre bien tout l'axe visible.
        all_vals = [
            v
            for vals in series.values()
            for v in vals
            if v > 0 and math.isfinite(v)
        ]

        y_min = min(all_vals) * 0.5 if all_vals else 1e-8
        y_max = max(all_vals) * 2.0 if all_vals else 1e2

        ax.set_yscale("log")

        # Zone vanishing (bas de graphe → seuil)
        ax.axhspan(
            y_min, vanishing_threshold,
            color="#3B8BD4", alpha=0.08, zorder=0,
            label=f"Vanishing (< {vanishing_threshold:.0e})"
        )
        ax.axhline(
            vanishing_threshold, color="#3B8BD4", linewidth=1.0,
            linestyle="--", alpha=0.7, zorder=1,
        )
        ax.text(
            epochs[-1] * 0.02, vanishing_threshold * 1.3,
            f"vanishing  {vanishing_threshold:.0e}",
            fontsize=9, color="#185FA5", va="bottom",
        )

        # Zone exploding (seuil → haut de graphe)
        ax.axhspan(
            exploding_threshold, y_max,
            color="#E24B4A", alpha=0.08, zorder=0,
            label=f"Exploding (> {exploding_threshold:.0e})"
        )
        ax.axhline(
            exploding_threshold, color="#E24B4A", linewidth=1.0,
            linestyle="--", alpha=0.7, zorder=1,
        )
        ax.text(
            epochs[-1] * 0.02, exploding_threshold * 0.75,
            f"exploding  {exploding_threshold:.0e}",
            fontsize=9, color="#A32D2D", va="top",
        )

        # ── courbes par couche ────────────────────────────────────────────────
        for name, values in series.items():
            label = name.replace(".weight", "")
            ax.plot(epochs, values, label=label, color=colors[name], linewidth=1.8, zorder=2)

        ax.set_ylim(y_min, y_max)
        ax.set_title("Norme L2 des gradients par couche", fontsize=13, pad=10)
        ax.set_ylabel("‖∇‖₂  (log scale)")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=8, ncol=2, loc="best")

        # ── overlay loss ──────────────────────────────────────────────────────
        if ax_loss is not None:
            loss_epochs = list(range(len(self.stats.loss)))
            ax_loss.semilogy(loss_epochs, self.stats.loss,
                             color="#e05c5c", linewidth=1.8, label="train loss")

            if self.stats.valLoss:
                val_epochs = list(range(len(self.stats.valLoss)))
                ax_loss.semilogy(val_epochs, self.stats.valLoss,
                                 color="#e09a5c", linewidth=1.8,
                                 linestyle="--", label="val loss")

            ax_loss.set_xlabel("Epoch")
            ax_loss.set_ylabel("Loss")
            ax_loss.grid(True, alpha=0.3, which="both")
            ax_loss.legend(fontsize=8, loc="best")
        else:
            ax.set_xlabel("Epoch")

        plt.tight_layout()
        plt.show()

    def plotActivations(
            self,
            stat: str = "norm",
            phases: list = None,
            dead_threshold: float = 0.3,
            overlay_loss: bool = True,
    ):
        """
        Trace les statistiques d'activations par couche au fil des epochs.

        Args:
            stat           : statistique à tracer parmi :
                             "norm"      — norme L2 normalisée  (défaut)
                             "mean"      — moyenne
                             "std"       — écart-type
                             "frac_neg"  — fraction d'activations négatives
                             "frac_dead" — fraction d'activations quasi-nulles
            phases         : liste de phases à afficher, ex. ["train"], ["val"],
                             ["train", "val"]. Par défaut toutes les phases présentes.
            dead_threshold : pour frac_dead/frac_neg, seuil d'alerte (défaut 0.3)
            overlay_loss   : affiche la train/val loss en sous-graphe
        """
        if not self.stats.activations:
            print("Aucune donnée d'activation enregistrée.")
            return

        STAT_META = {
            "norm": {"label": "Norme L2 normalisée des activations", "log": False, "alert": False},
            "mean": {"label": "Moyenne des activations", "log": False, "alert": False},
            "std": {"label": "Écart-type des activations", "log": False, "alert": False},
            "frac_neg": {"label": "Fraction d'activations négatives", "log": False, "alert": True},
            "frac_dead": {"label": "Fraction d'activations quasi-mortes", "log": False, "alert": True},
        }
        if stat not in STAT_META:
            raise ValueError(f"stat doit être l'un de : {list(STAT_META)}")

        meta = STAT_META[stat]

        # ── détermination des phases disponibles ──────────────────────────────
        all_phases = []
        for ep in self.stats.activations:
            for p in ep.keys():
                if p not in all_phases:
                    all_phases.append(p)

        if phases is None:
            phases = all_phases
        else:
            phases = [p for p in phases if p in all_phases]

        if not phases:
            print(f"Aucune des phases demandées n'est disponible. Phases présentes : {all_phases}")
            return

        # Style par phase : train = trait plein, val = pointillés
        PHASE_STYLE = {
            "train": {"linestyle": "-", "alpha": 1.0, "suffix": ""},
            "val": {"linestyle": "--", "alpha": 0.7, "suffix": " (val)"},
        }
        DEFAULT_STYLE = {"linestyle": "-", "alpha": 0.8, "suffix": ""}

        # ── noms de couches (union de toutes les phases/epochs) ───────────────
        layer_names = []
        for ep in self.stats.activations:
            for phase_data in ep.values():
                for name in phase_data:
                    if name not in layer_names:
                        layer_names.append(name)

        epochs = list(range(len(self.stats.activations)))
        n_layers = len(layer_names)
        cmap = plt.cm.get_cmap("tab20", n_layers)
        colors = {name: cmap(i) for i, name in enumerate(layer_names)}

        # ── layout ────────────────────────────────────────────────────────────
        if overlay_loss and self.stats.loss:
            fig, (ax, ax_loss) = plt.subplots(
                2, 1, figsize=(12, 9),
                sharex=True,
                gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
            )
        else:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax_loss = None

        if meta["log"]:
            ax.set_yscale("log")

        # ── zone d'alerte ─────────────────────────────────────────────────────
        if meta["alert"] and epochs:
            ax.axhspan(dead_threshold, 1.0, color="#E24B4A", alpha=0.08, zorder=0,
                       label=f"Alerte (> {dead_threshold:.0%})")
            ax.axhline(dead_threshold, color="#E24B4A", linewidth=1.0,
                       linestyle="--", alpha=0.7, zorder=1)
            ax.text(epochs[-1] * 0.02, dead_threshold * 1.05,
                    f"seuil  {dead_threshold:.0%}",
                    fontsize=9, color="#A32D2D", va="bottom")

        # ── courbes : une par (layer, phase) ──────────────────────────────────
        for phase in phases:
            style = PHASE_STYLE.get(phase, DEFAULT_STYLE)
            for name in layer_names:
                values = []
                for ep in self.stats.activations:
                    if phase in ep and name in ep[phase]:
                        values.append(ep[phase][name][stat])
                    else:
                        values.append(float("nan"))

                short = name.replace(".weight", "")
                label = f"{short}{style['suffix']}"
                ax.plot(
                    epochs, values,
                    label=label,
                    color=colors[name],
                    linewidth=1.8,
                    linestyle=style["linestyle"],
                    alpha=style["alpha"],
                    zorder=2,
                )

        if meta["alert"]:
            ax.set_ylim(0, 1.05)

        phase_str = " vs ".join(phases)
        ax.set_title(f"{meta['label']} — {phase_str}", fontsize=13, pad=10)
        ax.set_ylabel(meta["label"])
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=8, ncol=2, loc="best")

        # ── overlay loss ───────────────────────────────────────────────────────
        if ax_loss is not None:
            loss_epochs = list(range(len(self.stats.loss)))
            ax_loss.semilogy(loss_epochs, self.stats.loss,
                             color="#e05c5c", linewidth=1.8, label="train loss")
            if self.stats.valLoss:
                val_epochs = list(range(len(self.stats.valLoss)))
                ax_loss.semilogy(val_epochs, self.stats.valLoss,
                                 color="#e09a5c", linewidth=1.8,
                                 linestyle="--", label="val loss")
            ax_loss.set_xlabel("Epoch")
            ax_loss.set_ylabel("Loss")
            ax_loss.grid(True, alpha=0.3, which="both")
            ax_loss.legend(fontsize=8, loc="best")
        else:
            ax.set_xlabel("Epoch")

        plt.tight_layout()
        plt.show()

    def plotPerOutputGrads(
            self,
            layer: str = None,
            vanishing_threshold: float = 1e-4,
            exploding_threshold: float = 10.0,
            overlay_loss: bool = True,
            output_names: list[str] = None,  # ← ex: ["vx", "vy", "vz"]
    ):
        if not hasattr(self.stats, "perOutputGrads") or not self.stats.perOutputGrads:
            print("Aucune donnée perOutputGrads enregistrée.")
            return

        epochs = list(range(len(self.stats.perOutputGrads)))
        all_layers = list(self.stats.perOutputGrads[0][0].keys())
        n_out = len(self.stats.perOutputGrads[0])

        if output_names is not None and len(output_names) != n_out:
            raise ValueError(
                f"output_names a {len(output_names)} entrées mais le modèle a {n_out} sorties."
            )
        labels = output_names if output_names is not None else [f"out {i}" for i in range(n_out)]

        if layer is not None:
            if layer not in all_layers:
                raise ValueError(f"Couche '{layer}' introuvable. Disponibles : {all_layers}")
            layers_to_plot = [layer]
        else:
            layers_to_plot = all_layers

        cmap = plt.cm.get_cmap("tab20", n_out)
        colors = [cmap(i) for i in range(n_out)]
        linestyles = ["-", "--", "-."]  # tourne tous les 3 outputs

        for lname in layers_to_plot:

            if overlay_loss and self.stats.loss:
                fig, (ax, ax_loss) = plt.subplots(
                    2, 1, figsize=(12, 9),
                    sharex=True,
                    gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
                )
            else:
                fig, ax = plt.subplots(figsize=(12, 6))
                ax_loss = None

            series = {
                i: [
                    self.stats.perOutputGrads[e][i][lname]["grad_norm"]
                    if i in self.stats.perOutputGrads[e]
                       and lname in self.stats.perOutputGrads[e][i]
                    else float("nan")
                    for e in epochs
                ]
                for i in range(n_out)
            }

            all_vals = [v for s in series.values() for v in s if v == v and v > 0]
            y_min = min(all_vals) * 0.5 if all_vals else 1e-8
            y_max = max(all_vals) * 2.0 if all_vals else 1e2

            ax.set_yscale("log")
            ax.set_ylim(y_min, y_max)

            ax.axhspan(y_min, vanishing_threshold, color="#3B8BD4", alpha=0.08, zorder=0)
            ax.axhline(vanishing_threshold, color="#3B8BD4", linewidth=1.0,
                       linestyle="--", alpha=0.7)
            ax.text(epochs[-1] * 0.02, vanishing_threshold * 1.3,
                    f"vanishing  {vanishing_threshold:.0e}",
                    fontsize=9, color="#185FA5", va="bottom")

            ax.axhspan(exploding_threshold, y_max, color="#E24B4A", alpha=0.08, zorder=0)
            ax.axhline(exploding_threshold, color="#E24B4A", linewidth=1.0,
                       linestyle="--", alpha=0.7)
            ax.text(epochs[-1] * 0.02, exploding_threshold * 0.75,
                    f"exploding  {exploding_threshold:.0e}",
                    fontsize=9, color="#A32D2D", va="top")

            for i, vals in series.items():
                ax.plot(
                    epochs, vals,
                    label=labels[i],
                    color=colors[i],
                    linestyle=linestyles[i % len(linestyles)],
                    linewidth=1.6,
                    zorder=2,
                )

            short = lname.replace(".weight", "")
            ax.set_title(f"‖dL_i/dθ‖ — couche {short}", fontsize=13, pad=10)
            ax.set_ylabel("‖∇‖₂  (log scale)")
            ax.grid(True, alpha=0.3, which="both")
            ax.legend(fontsize=8, ncol=2, loc="best")

            if ax_loss is not None:
                loss_epochs = list(range(len(self.stats.loss)))
                ax_loss.semilogy(loss_epochs, self.stats.loss,
                                 color="#e05c5c", linewidth=1.8, label="train loss")
                if self.stats.valLoss:
                    ax_loss.semilogy(list(range(len(self.stats.valLoss))),
                                     self.stats.valLoss,
                                     color="#e09a5c", linewidth=1.8,
                                     linestyle="--", label="val loss")
                ax_loss.set_xlabel("Epoch")
                ax_loss.set_ylabel("Loss")
                ax_loss.grid(True, alpha=0.3, which="both")
                ax_loss.legend(fontsize=8, loc="best")
            else:
                ax.set_xlabel("Epoch")

            plt.tight_layout()
            plt.show()


    def plotLoss(self):
        plt.semilogy(self.stats.loss, label='train')
        plt.semilogy(self.stats.valLoss, label='val')
        plt.legend()
        plt.show()

    def plotLr(self):
        plt.plot(self.stats.lr)
        plt.show()

    @staticmethod
    def _format_loss(val):
        return f"{val:.3e}" if val < 1e-3 else f"{val:.6f}"