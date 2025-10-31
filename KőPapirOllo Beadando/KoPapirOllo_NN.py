import random

# Lehetséges lépések
lepesek = ["ko", "papir", "ollo"]

# Statisztika a játékos eddigi lépéseiről
jatekos_tortenet = []

def gep_tippel():
    """Az AI megpróbálja kitalálni, mit lépsz legközelebb"""
    if len(jatekos_tortenet) < 3:
        # Ha nincs elég adat, tippel véletlenszerűen
        return random.choice(lepesek)

    # Az utolsó 2 lépés alapján próbál mintát találni
    minta = tuple(jatekos_tortenet[-2:])
    # Megszámolja, mi következett leggyakrabban e minta után
    stat = {}
    for i in range(len(jatekos_tortenet) - 2):
        if tuple(jatekos_tortenet[i:i+2]) == minta:
            kov = jatekos_tortenet[i+2]
            stat[kov] = stat.get(kov, 0) + 1

    if not stat:
        # Ha nem talál mintát, véletlenszerű
        return random.choice(lepesek)

    # Leggyakoribb folytatás
    tipp = max(stat, key=stat.get)

    # AI az ellened legjobb lépést választja
    if tipp == "ko":
        return "papir"   # Papír veri a követ
    elif tipp == "papir":
        return "ollo"    # Olló veri a papírt
    else:
        return "ko"      # Kő veri az ollót

def eredmeny(jatekos, gep):
    """Visszaadja a kör eredményét"""
    if jatekos == gep:
        return "Döntetlen!"
    elif (jatekos == "ko" and gep == "ollo") or \
         (jatekos == "papir" and gep == "ko") or \
         (jatekos == "ollo" and gep == "papir"):
        return "Nyertél!"
    else:
        return "A gép nyert!"

# Fő ciklus
jatekos_pont = 0
gep_pont = 0

print("Kő–Papír–Olló játék (írj 'kilep' ha vége)!")
while True:
    jatekos = input("Válassz (ko/papir/ollo): ").strip().lower()
    if jatekos == "kilep":
        break
    if jatekos not in lepesek:
        print("Hibás bemenet, próbáld újra!")
        continue

    gep = gep_tippel()
    print(f"A gép választása: {gep}")
    eredm = eredmeny(jatekos, gep)
    print(eredm)

    if "Nyertél" in eredm:
        jatekos_pont += 1
    elif "gép nyert" in eredm:
        gep_pont += 1

    jatekos_tortenet.append(jatekos)
    print(f"Eredmény: Te {jatekos_pont} - {gep_pont} Gép\n")

print("\nJáték vége!")
print(f"Végső eredmény: Te {jatekos_pont} - {gep_pont} Gép")
