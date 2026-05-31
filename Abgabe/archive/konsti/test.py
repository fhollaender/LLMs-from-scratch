import statistics

messwerte = [12.4, None, 7.8, 99.9, 5.1, None, 8.3, 102.5, 6.7, None, 9.2]

def bereinigte_daten(werte: list) -> list:
    bereinigt = []

    for wert in werte:
        if wert is not None and wert <= 50.0:
            bereinigt.append(wert)
    
    return bereinigt  # nach der Schleife!

aufgabe_eins = bereinigte_daten(messwerte)
# → [12.4, 7.8, 5.1, 8.3, 6.7, 9.2]
def berechne_statistik(werte: list) -> tuple:
    if werte:
        
        minimum = min(werte)
        maximum = max(werte)
        median = statistics.median(werte)

    return minimum, maximum, median

aufgabe_zwei = berechne_statistik(aufgabe_eins)

def erstelle_bericht(werte: list) -> None:
    sauber = bereinigte_daten(werte)               
    min, max, median = berechne_statistik(sauber)
    print(f"Bereinigte Daten: {sauber}")
    print(f"Minimum: {min}")
    print(f"Maximum: {max}")
    print(f"Median: {median}")

aufgabe_drei = erstelle_bericht(messwerte)
print(aufgabe_drei)


noten = [1.0, 2.3, 1.7, 3.5, 4.1, 2.0, 5.0, 1.3, 3.8, 2.7]

def berechne_noten(werte: list) -> dict:
    noten_bild = {'sehr gut': 0,'gut': 0, 'befriedigend' : 0, 'ausreichend' : 0, 'nicht bestanden' : 0 }

    for wert in werte:
        if wert < 2.0:
            noten_bild['sehr gut'] += 1
        elif wert < 3.0:
            noten_bild['gut'] += 1
        elif wert < 4.0:
            noten_bild['befriedigend'] += 1
        elif wert < 5.0:
            noten_bild['ausreichend'] += 1
        elif wert < 6.0:
            noten_bild['nicht bestanden'] += 1
    return noten_bild

aufgabe_vier = berechne_noten(noten)
print(aufgabe_vier)



puzahlen = [3, 7, 2, 9, 4, 6, 1, 8, 5, 10]

def analysieren(werte: list) -> dict:

    dectionary = {'gerade': [], 'ungerade': [], 'summe_gerade': 0, 'summe_ungerade': 0}

    for wert in werte:
        if wert % 2 == 0:
            
            dectionary['gerade'].append(wert) 
            dectionary['summe_gerade'] += wert
        else:
          
            dectionary['ungerade'].append(wert)
            dectionary['summe_ungerade'] += wert
    return dectionary

Auswertung_der_Analyse = analysieren(puzahlen)
#print(Auswertung_der_Analyse)

 
zahlen = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

def nur_geraden(wert: list) -> list:
    deconary = {'nur gerade': [], 'summme der geraden': 0}
    for werte in wert:
        if werte % 2 == 0:
            deconary['nur gerade'].append(werte)
            deconary['summme der geraden'] += werte
    return deconary

GeradeAusgabe = nur_geraden(zahlen)


