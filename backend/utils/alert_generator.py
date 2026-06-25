class AlertGenerator:
    @staticmethod
    def generate(rul: float, failure_probability: float):
        if rul <= 10:
            return ("FAILURE_IMMINENT",'Engine predicted to fail very soon.')
        
        elif (failure_probability >= 0.80 or rul <= 30):
            return ("CRITICAL","Immediate maintenance recommended.")

        elif (failure_probability >= 0.50 or rul <= 60):
            return ("WARNING","Monitor Engine closely")
        
        return ("SAFE","Engine operating normally.")