import random
import time

lepesek = ["ko", "papir", "ollo"]

ascii_art = {
    "ko": """
        _______
    ---'   ____)
          (_____)
          (_____)
          (____)
    ---.__(___)
    """,
    "papir": """
         _______
    ---'    ____)____
               ______)
              _______)
             _______)
    ---.__________)
    """,
    "ollo": """
        _______
    ---'   ____)____
              ______)
           __________)
          (____)
    ---.__(___)
    """
}

jatekos_tortenet = []

def gep_tippel():
    if len(jatekos_tortenet) < 3:
        return random.choice(lepesek)
    minta = tuple(jatekos_tortenet[-2:])
    stat = {}
    for i in range(len(jatekos_tortenet) - 2):
        if tuple(jatekos_tortenet[i:i+2]) == minta:
            kov = jatekos_tortenet[i+2]
            stat[kov] = stat.get(kov, 0) + 1
    if not stat:
        return random.choice(lepesek)
    tipp = max(stat, key=stat.get)
    if tipp == "ko":
        return "papir"
    elif tipp == "papir":
        return "ollo"
    else:
        return "ko"

def eredmeny(jatekos, gep):
    if jatekos == gep:
        return "Döntetlen!"
    elif (jatekos == "ko" and gep == "ollo") or \
         (jatekos == "papir" and gep == "ko") or \
         (jatekos == "ollo" and gep == "papir"):
        return "Nyertél!"
    else:
        return "A gép nyert!"

jatekos_pont = 0
gep_pont = 0

print("Tanulo Ko-Papir-Ollo jatek")
print("Irj 'kilep' ha be akarod fejezni.\n")

while True:
    jatekos = input("Valassz (ko/papir/ollo): ").strip().lower()
    if jatekos == "kilep":
        break
    if jatekos not in lepesek:
        print("Hibas bemenet, probald ujra!")
        continue

    gep = gep_tippel()

    print("\nTe valasztottad:")
    print(ascii_art[jatekos])
    time.sleep(0.5)
    print("A gep valasztasa:")
    print(ascii_art[gep])
    time.sleep(0.5)

    eredm = eredmeny(jatekos, gep)
    print(f"Eredmeny: {eredm}\n")

    if "Nyertél" in eredm:
        jatekos_pont += 1
    elif "gép nyert" in eredm:
        gep_pont += 1

    jatekos_tortenet.append(jatekos)

    print(f"Allas: Te {jatekos_pont} - {gep_pont} Gep\n")
    print("--------------------------------------------------")

print("\nJatek vege!")
print(f"Vegso eredmeny: Te {jatekos_pont} - {gep_pont} Gep")

if jatekos_pont > gep_pont:
    print("Gratulalok, legyozted a tanulo AI-t!")
elif jatekos_pont < gep_pont:
    print("A gep gyozott! Lehet, hogy kiismerte a taktikadat...")
else:
    print("Dontetlen merkozes!")
