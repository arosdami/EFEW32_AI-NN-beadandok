import random

m = random.random()
b = random.random()
tanulas = 0.01

igazi_m = 2
igazi_b = 1

adatok = []
for i in range(20):
    x = random.uniform(0, 10)
    y = igazi_m * x + igazi_b
    adatok.append((x, y))

for lepes in range(100):
    x, y = random.choice(adatok)
    
    tipp = m * x + b
    hiba = (tipp - y) ** 2
    
    dH_dm = 2 * (tipp - y) * x
    dH_db = 2 * (tipp - y)
    
    m = m - tanulas * dH_dm
    b = b - tanulas * dH_db
    
    if lepes < 10 or lepes % 10 == 0:
        print(f"{lepes}. hiba={hiba:.4f} m={m:.4f} b={b:.4f}")

print(f"\nKesz: m={m:.4f} b={b:.4f}")
print(f"Igazi: m={igazi_m} b={igazi_b}")