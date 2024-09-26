class StepBetaScheduler:
    def __init__(self, flag, T_max, end_value=1):
        """
        Set T_max to the maximum epoch
        """
        self.T_max = T_max
        self.end_value = end_value
        self.current_step = 0
        self.beta = 0.0  # Initial value of beta
        self.step_value = (self.end_value - self.beta) / self.T_max
        self.flag = flag

    def step(self):
        """Get the current beta value and update the step."""
        if self.flag == False:
            return self.end_value

        # Update the current step
        self.beta += self.step_value
        return self.beta
