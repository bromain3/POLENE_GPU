
class TrainerStats:
    def __init__(self):
        self.loss = []
        self.valLoss = []
        self.lr = []

class _statsSave:
    def __init__(self):
        self.stats = TrainerStats()

    @staticmethod
    def _format_loss(val):
        return f"{val:.3e}" if val < 1e-3 else f"{val:.6f}"