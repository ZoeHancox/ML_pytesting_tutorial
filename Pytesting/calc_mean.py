def calculate_mean(numbers):
    if len(numbers) == 0:
        return None
    else:
        return sum(numbers) / len(numbers)