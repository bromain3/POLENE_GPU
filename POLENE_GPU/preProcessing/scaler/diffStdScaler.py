from POLENE_GPU.preProcessing.scaler.stdScaler import stdScaler
from POLENE_GPU.dataHandler.serie import serie

import torch


class diffStdScaler(stdScaler):
    """
    Scaler dédié à la prédiction d'incréments Y_{t+dt} - Y_t.

    Contrairement à `stdScaler`, où l'on appelle `.fit(x)` après construction,
    ici `inputs` (Y_t) et `targets` (Y_{t+dt}) sont fournis DIRECTEMENT AU
    CONSTRUCTEUR, et les statistiques (mean, std, is_constant) sont calculées
    immédiatement. `fit()` devient donc un no-op, conservé uniquement pour
    compatibilité avec les pipelines qui appellent `.fit(x)` avant `.transform(x)`.

    Option S2
    ---------
    Si un scaler secondaire déjà fit `S2` est fourni :
        - les statistiques sont calculées sur S2(targets - inputs) et non sur
          l'incrément brut (targets - inputs). Utile quand Y est normalement
          scalé d'une façon particulière (log, scaler physique, par espèce...)
          et qu'on veut que l'échelle de l'incrément soit cohérente avec ça
          plutôt qu'avec l'échelle brute (ex: variance de l'incrément de N2
          écrasée par la variance brute inter-échantillons).
        - `transform(diff)` applique S2 PUIS standardise avec ces stats.
        - `inverse_transform(z)` déstandardise PUIS applique S2.inverse_transform.

    Note sur J / J_1
    ----------------
    `J`/`J_1` (hérités de stdScaler) donnent le Jacobien de la seule étape de
    standardisation, par rapport à S2(diff) (pas par rapport au diff brut).
    Si tu as besoin du Jacobien total par rapport au diff brut et que S2 est
    non trivial, il faut le composer manuellement :
        J_total(diff) = J(S2(diff)) * S2.J(diff)   (règle de la chaîne)
    """

    def __init__(self, inputs, targets, S2=None, varianceCutoff=1e-28, checkShapes=True):
        super().__init__(varianceCutoff=varianceCutoff)
        self.S2 = S2

        diff = self.__computeDiffSerie(inputs, targets, checkShapes)

        statData = diff if self.S2 is None else self.S2.transform(diff)

        # Réutilise stdScaler.fit() tel quel pour calculer mean/std/is_constant
        # (et _input_ndim) sur statData.
        super().fit(statData)

    # ------------------------------------------------------------------
    # helpers internes
    # ------------------------------------------------------------------

    def __computeDiffSerie(self, inputs, targets, checkShapes):
        if checkShapes and inputs.data.shape != targets.data.shape:
            raise ValueError(
                f"inputs et targets doivent avoir la même shape pour calculer un "
                f"incrément (reçu {tuple(inputs.data.shape)} vs {tuple(targets.data.shape)})."
            )

        diffData = targets.data - inputs.data
        return serie(diffData, targets.colName,
                     storingDevice=targets.storingDevice,
                     storingDtype=targets.storingDtype)

    # ------------------------------------------------------------------
    # fit : no-op, tout a déjà été calculé au constructeur
    # ------------------------------------------------------------------

    def fit(self, x=None):
        return self

    # fit_transform est hérité de stdScaler : il appelle self.fit(x) (no-op
    # ici) puis self.transform(x) (notre version, S2-aware) -> fonctionne
    # correctement par polymorphisme, pas besoin de le redéfinir.

    # ------------------------------------------------------------------
    # transform / inverse_transform : on compose S2 autour de la
    # standardisation déjà implémentée dans stdScaler
    # ------------------------------------------------------------------

    def transform(self, diff):
        """
        diff : serie représentant l'incrément BRUT Y_{t+dt} - Y_t
               (pas encore passé par S2).
        """
        x = diff if self.S2 is None else self.S2.transform(diff)
        return super().transform(x)

    def inverse_transform(self, z):
        """
        Retourne l'incrément dans l'espace BRUT (avant S2), à partir de sa
        version standardisée z.
        """
        unscaled = super().inverse_transform(z)
        if self.S2 is not None:
            unscaled = self.S2.inverse_transform(unscaled)
        return unscaled
