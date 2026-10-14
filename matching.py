from datetime import datetime


def matching_score(
    donor_location,
    ngo_location,
    available_kg,
    needed_kg,
    pickup_deadline
):
    score = 0

    if donor_location.strip().lower() == ngo_location.strip().lower():
        score += 40

    if needed_kg > 0 and available_kg >= needed_kg:
        score += 30
    elif needed_kg > 0 and available_kg > 0:
        score += 15

    try:
        deadline = datetime.fromisoformat(pickup_deadline)
        hours_left = (
            deadline - datetime.now()
        ).total_seconds() / 3600

        if 0 < hours_left <= 4:
            score += 30
        elif 4 < hours_left <= 12:
            score += 20
        elif hours_left > 12:
            score += 10
    except ValueError:
        pass

    return score