import winsound

def trigger_alert(prob):
    if prob > 0.6:
        print("🚨 HIGH ALERT")
        winsound.Beep(2000, 500)

    elif prob > 0.3:
        print("⚠️ Moderate Warning")
        winsound.Beep(1000, 300)