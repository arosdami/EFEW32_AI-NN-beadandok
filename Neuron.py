import random

suly = 1.2
eltolas = 19.9

tanito_adatok = [
    (20, 100),
    (40, 80), 
    (60, 60),
    (80, 40),
    (100, 20)
]

for lepes in range(100):
    energy, actual_distance = random.choice(tanito_adatok)
    predicted_distance = suly * energy + eltolas
    error = predicted_distance - actual_distance
    
    if error > 0:
        suly -= 0.1
        eltolas -= 0.1
    else:
        suly += 0.1
        eltolas += 0.1
    
    if lepes % 20 == 0:
        print(f"Lepes {lepes}: W={suly:.2f}, B={eltolas:.2f}, Error={error:.2f}")

print(f"\nVegeredmeny: Weight={suly:.2f}, Bias={eltolas:.2f}")

print(f"\n--- TESZTELES ---")
for energy, actual_distance in tanito_adatok:
    predicted = suly * energy + eltolas
    error = predicted - actual_distance
    print(f"Energy={energy}: Predicted={predicted:.1f}, Actual={actual_distance}, Error={error:.1f}")