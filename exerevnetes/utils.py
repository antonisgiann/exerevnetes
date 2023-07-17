def time_format(t):
    hours = int(t//3600)
    minutes = int((t - hours * 3600) // 60)
    seconds = int(t - (hours * 3600 + minutes * 60))
    
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"