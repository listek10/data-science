# Lekcja 01 – Data Science i Machine Learning
## Odpowiedzi do zadań 8–20

---

## Zadanie 8 – Przykłady zastosowań DS dla ekologii

### Projekt 1: Prognoza produkcji energii z farmy wiatrowej
- **Problem:** operator sieci nie wie z wyprzedzeniem, ile prądu dadzą turbiny, więc na wszelki wypadek trzyma w rezerwie bloki węglowe.
- **Typ zadania ML:** regresja (prognoza szeregu czasowego).
- **Dane:** prognozy pogody (prędkość i kierunek wiatru, ciśnienie, temperatura), historyczna produkcja każdej turbiny, dane techniczne turbin, godzina i pora roku.
- **Wpływ na środowisko:** mniejsza rezerwa z elektrowni węglowych i mniej marnowanej energii odnawialnej.

### Projekt 2: Wykrywanie wylesiania ze zdjęć satelitarnych
- **Problem:** nielegalne wycinki są wykrywane za późno, często po miesiącach.
- **Typ zadania ML:** segmentacja / klasyfikacja obrazów (detekcja zmian między datami).
- **Dane:** zdjęcia satelitarne (Sentinel-2, Landsat) z kolejnych tygodni, mapy obszarów chronionych, historyczne zgłoszenia wycinek.
- **Wpływ na środowisko:** reakcja służb w dni zamiast w miesiące, ochrona lasów pochłaniających CO2.

### Projekt 3: Optymalizacja wywozu odpadów
- **Problem:** śmieciarki jeżdżą stałą trasą i opróżniają w połowie puste kontenery, spalając paliwo.
- **Typ zadania ML:** regresja (prognoza zapełnienia kontenera) + optymalizacja trasy.
- **Dane:** czujniki poziomu zapełnienia, historia wywozów, kalendarz (święta, weekendy), GPS pojazdów, mapa drogowa.
- **Wpływ na środowisko:** mniej przejechanych kilometrów, mniej spalin i hałasu w mieście.

---

## Zadanie 9 – Projektowanie systemu rekomendacji

### Typ zadania ML
**Ranking** – układamy listę filmów od najbardziej do najmniej trafnego. Pod spodem działa model predykcyjny: regresja (jaka będzie ocena) lub klasyfikacja (czy użytkownik obejrzy do końca).

### Potrzebne dane
- **Użytkownik:** historia oglądania, oceny, język, kraj, urządzenie.
- **Film:** gatunek, rok, obsada, długość, opis, popularność.
- **Interakcje:** co obejrzał, ile minut, co porzucił po 5 minutach, w co kliknął i nie obejrzał.
- **Kontekst:** pora dnia, dzień tygodnia, urządzenie.

Najcenniejsze są interakcje – mówią więcej niż deklarowane preferencje.

### Podejście do uczenia
**Supervised, hybrydowe:**
- *Collaborative filtering* – "użytkownicy podobni do Ciebie obejrzeli X". Daje najlepszą trafność, ale nie działa dla nowych filmów i nowych użytkowników (problem zimnego startu).
- *Content-based* – podobieństwo cech filmu. Ratuje zimny start.

Łączymy oba: content-based dla nowości, collaborative dla reszty.

### Metryka oceny
- **Offline:** NDCG@10 i Recall@10 – czy trafione pozycje są wysoko na liście.
- **Online (ważniejsze):** test A/B – czas oglądania, odsetek kliknięć, utrzymanie subskrypcji.

### Optymalizacja Green AI
1. **Dwuetapowy ranking:** tani model wybiera 500 kandydatów z katalogu, drogi model ocenia tylko te 500.
2. **Przeliczanie w batchu:** profile i embeddingi liczone raz na dobę, nie przy każdym wejściu.
3. **Cache:** gotowa lista rekomendacji ważna kilka godzin.
4. **Retrening tylko przy spadku jakości**, nie codziennie z przyzwyczajenia.
5. **Trening w regionie chmury z zieloną energią.**

---

## Zadanie 10 – Analiza trade-off: dokładność vs energia

### Efficiency ratio (accuracy / Wh)

| Model | Accuracy | Energia | Efficiency ratio |
|---|---|---|---|
| Logistic Regression | 75% | 0,5 Wh | **150,0** |
| Random Forest | 85% | 5 Wh | **17,0** |
| EfficientNet | 90% | 50 Wh | **1,8** |
| ResNet50 | 92% | 500 Wh | **0,18** |

### Wykres konceptualny

```
accuracy
  92% |                                      ResNet50 ●
  90% |                        EfficientNet ●
  85% |          RandomForest ●
  75% |  LogReg ●
      +------------------------------------------------
        0,5 Wh      5 Wh         50 Wh        500 Wh   (skala logarytmiczna)
```

Krzywa się wypłaszcza: **ostatnie 2 punkty procentowe kosztują 10x więcej energii niż całe poprzednie 15 punktów.** To klasyczne prawo malejących przychodów.

### Kiedy który model
- **Logistic Regression** – baseline na start projektu, ogromna skala, bardzo tanie urządzenia.
- **Random Forest** – dane tabelaryczne, średnia skala, gdy potrzebna wyjaśnialność.
- **EfficientNet** – domyślny wybór produkcyjny, najlepszy kompromis.
- **ResNet50** – gdy błąd kosztuje więcej niż energia.

### Aplikacja mobilna → EfficientNet
Został zaprojektowany pod urządzenia o ograniczonych zasobach. Daje 90% przy 10x mniejszym zużyciu niż ResNet50. Na telefonie liczy się bateria i czas odpowiedzi, a 2 punkty procentowe nie są warte rozładowanego telefonu.

### System bezpieczeństwa w elektrowni jądrowej → ResNet50
Koszt błędu jest nieporównywalny z kosztem energii. 500 Wh to ułamek grosza, a przeoczona usterka to katastrofa. Zasilanie jest stałe, urządzenie nie działa na baterii. Dodatkowo: kilka modeli głosujących i obowiązkowe potwierdzenie przez człowieka.

---

## Zadanie 11 – Design case: wykrywanie fraudu

### Typ zadania ML
**Klasyfikacja binarna** (fraud / nie fraud), w praktyce scoring: model zwraca prawdopodobieństwo, a próg ustawiamy biznesowo.

### Supervised czy unsupervised? – Oba
- **Supervised** jako podstawa: mamy etykiety z reklamacji i chargebacków. Model uczy się znanych schematów oszustw.
- **Unsupervised** (detekcja anomalii) jako uzupełnienie: wyłapuje **nowe** schematy, których nie ma w danych historycznych. Oszuści ciągle zmieniają metody, więc sam supervised zawsze będzie spóźniony.

### Wyzwania
| Wyzwanie | Rozwiązanie |
|---|---|
| Imbalanced data (0,1%) | Ważenie klas, undersampling większości, metryki inne niż accuracy |
| Opóźnione etykiety (chargeback po tygodniach) | Trenowanie na danych sprzed kilku tygodni, osobny model na świeże anomalie |
| Concept drift | Monitoring jakości, retrening przy spadku |
| Limit czasu < 100 ms | Lekki model, cechy liczone wcześniej |
| Koszt FP vs FN | Próg dobrany pod koszt w złotówkach, nie pod F1 |

### Ocena modelu
**Nie accuracy** – model mówiący zawsze "nie fraud" ma 99,9% accuracy i jest bezużyteczny.

Używamy: **precision, recall, PR-AUC**. Kluczowa jest też metryka operacyjna: **recall przy limicie alertów**, który zespół jest w stanie obsłużyć dziennie. Próg wybieramy tam, gdzie koszt zablokowanych uczciwych klientów równoważy się ze stratami z oszustw.

### Energooszczędność przy 10 mln transakcji dziennie
1. **Kaskada:** proste reguły odsiewają ~99% oczywiście poprawnych transakcji; model widzi tylko 1%. To redukuje koszt stukrotnie.
2. **Lekki model** (gradient boosting) zamiast sieci głębokiej – przy danych tabelarycznych daje porównywalną jakość za ułamek energii.
3. **Feature store:** cechy klienta liczone raz i współdzielone, nie przeliczane przy każdej transakcji.
4. **Inference na CPU**, bez GPU.
5. **Retrening tygodniowy lub wyzwalany driftem**, nie codzienny.

---

## Zadanie 12 – Comparison: reguły vs ML w moderacji komentarzy

### Rozwiązanie oparte na regułach
Lista słów zabronionych, wyrażenia regularne (także na próby obejścia typu "k*rwa"), limity: dużo wielkich liter, linki, powtórzenia, czarna lista użytkowników.

### Rozwiązanie oparte na ML
Klasyfikator tekstu uczony na oznaczonych komentarzach. Wersja tania: TF-IDF + regresja logistyczna. Wersja dokładna: dostrojony model transformerowy.

### Tabela porównawcza

| Kryterium | Reguły | Machine Learning |
|---|---|---|
| Dokładność | 60–70% | 85–93% |
| Rozumienie kontekstu | brak (blokuje "nie bądź dupkiem" i cytaty) | dobre |
| Ironia, obraza bez wulgaryzmów | nie wykrywa | wykrywa częściowo |
| Łatwość utrzymania | trudna – lista rośnie w nieskończoność | łatwiejsza – dodajemy przykłady, nie reguły |
| Zużycie energii | praktycznie zerowe | trening + koszt każdej predykcji |
| Koszt wdrożenia | niski | wyższy (dane, etykietowanie, infrastruktura) |
| Wyjaśnialność | pełna ("słowo X z listy") | ograniczona |
| Szybkość reakcji na nowe zjawisko | natychmiastowa | wymaga danych i retreningu |

### Co jest lepsze?
**ML wygrywa na dokładności, ale sam w sobie nie jest najlepszym wyborem.** Reguły są bezkonkurencyjne na oczywistych przypadkach i nic nie kosztują.

### Rozwiązanie hybrydowe (rekomendowane)
1. **Warstwa 1 – reguły:** oczywiste wulgaryzmy i spam blokowane natychmiast, bez uruchamiania modelu. Odsiewa znaczną część ruchu za darmo.
2. **Warstwa 2 – lekki model ML:** ocenia resztę i zwraca prawdopodobieństwo toksyczności.
3. **Warstwa 3 – człowiek:** przypadki niepewne (wynik 0,4–0,6) trafiają do moderatora. Jego decyzje wracają jako nowe dane treningowe.

Taki układ daje dokładność ML przy znacznie niższym koszcie energii i zachowuje kontrolę człowieka tam, gdzie jest potrzebna.

---

## Zadanie 13 – Design end-to-end: Smart City Traffic

### Potrzebne dane
- Czujniki ruchu: pętle indukcyjne, kamery (liczba pojazdów, długość kolejek).
- Dane GPS z nawigacji i z floty komunikacji miejskiej (prędkości, czasy przejazdu).
- Aktualny stan i program sygnalizacji.
- Pogoda, kalendarz (święta, mecze, remonty), zdarzenia drogowe.
- Mapa sieci drogowej z geometrią skrzyżowań.
- Pomiary jakości powietrza (do oceny efektu).

### Typy zadań ML (jest ich kilka)
1. **Detekcja obiektów** – zliczanie pojazdów, pieszych i rowerzystów z kamer.
2. **Regresja / szeregi czasowe** – prognoza natężenia ruchu na najbliższe 15–60 minut.
3. **Klasyfikacja** – wykrywanie zdarzeń: wypadek, blokada, pojazd uprzywilejowany.
4. **Reinforcement learning** – sterowanie fazami świateł (zadanie główne).
5. **Klasteryzacja** – grupowanie typowych wzorców dni i skrzyżowań, osobne strategie dla każdego typu.

### Podejścia do uczenia
- Supervised dla detekcji i prognoz.
- RL dla sterowania – uczone **najpierw w symulatorze** (np. SUMO), bo eksperymentowanie na żywym mieście jest niedopuszczalne. Wdrożenie z twardymi ograniczeniami bezpieczeństwa (minimalny czas zielonego, maksymalny czas czerwonego).

### Pomiar sukcesu
- **Techniczne:** średni czas przejazdu korytarzem, liczba zatrzymań, długość kolejek.
- **Biznesowe / miejskie:** emisja CO2 i NOx, zużycie paliwa, czas przejazdu karetek i straży, liczba wypadków, czas oczekiwania pieszych.
- **Metodyka:** test A/B – skrzyżowania z systemem vs porównywalne skrzyżowania bez niego, w tym samym okresie.

### Green AI
- Trening agenta RL w symulatorze – tanio, bez zużycia paliwa przez prawdziwe auta.
- **Przetwarzanie obrazu lokalnie na sterowniku** (edge). Do centrum wysyłamy liczby, nie strumień wideo. To największa oszczędność: transmisja i przetwarzanie wideo w chmurze są wielokrotnie droższe.
- Małe, skwantyzowane modele wizyjne na urządzeniach.
- Retrening sezonowy, nie ciągły.
- **Bilans netto:** system zużywa energię, ale oszczędza paliwo tysięcy samochodów dziennie. Ten bilans trzeba policzyć i raportować – to uzasadnienie projektu.

### Ryzyka i wyzwania etyczne
- **Prywatność:** kamery rejestrują ludzi i tablice. Rozwiązanie: anonimizacja na urządzeniu, zapisywanie tylko zliczeń, brak przechowywania nagrań.
- **Sprawiedliwość:** optymalizacja "przepustowości" potrafi faworyzować arterie przelotowe kosztem osiedli i pieszych. Trzeba wprost nagradzać krótki czas oczekiwania pieszych i rowerzystów.
- **Bezpieczeństwo:** przy awarii lub dziwnym zachowaniu agenta automatyczne przejście na stały, sprawdzony program.
- **Cyberbezpieczeństwo:** sygnalizacja to infrastruktura krytyczna.
- **Odpowiedzialność:** kto odpowiada za wypadek przy sterowaniu automatycznym – miasto musi to ustalić przed wdrożeniem.

---

## Zadanie 14 – Semi-Supervised Strategy

### Jak wykorzystać 1 000 opisanych zdjęć
- Podzielić: ~700 trening, ~150 walidacja, ~150 test odłożony na sam koniec. **Zbiór testowy musi zostać nietknięty** – bez niego nie da się uczciwie ocenić efektu.
- Dostroić gotowy model (transfer learning z ResNet lub CLIP) zamiast trenować od zera.

### Jak wykorzystać 99 000 nieopisanych zdjęć
1. **Self-supervised pretraining** (np. SimCLR, DINO): model uczy się rozpoznawać, że dwa wycinki tego samego zdjęcia to to samo. Nie potrzebuje etykiet, a uczy się dobrych reprezentacji butów, koszul i spodni.
2. **Pseudo-labeling:** model bazowy etykietuje nieopisane zdjęcia; bierzemy tylko te z pewnością powyżej 0,95, dokładamy do treningu, powtarzamy 2–3 razy.
3. **Active learning:** wybieramy 500 zdjęć, co do których model jest najbardziej niepewny, i dajemy je człowiekowi do opisania. To najwyższy zysk jakości na jedną etykietę.

### Konkretna metodologia (krok po kroku)
1. DINO na wszystkich 100 000 zdjęć (bez etykiet) → dobry backbone.
2. Fine-tuning na 700 opisanych zdjęciach.
3. Pseudo-labeling z progiem pewności, 2–3 iteracje.
4. Active learning: 500 najtrudniejszych zdjęć do opisania przez człowieka.
5. Finalny fine-tuning.

### Porównanie z baseline
Ten sam odłożony zbiór testowy, ta sama metryka. Baseline = fine-tuning tylko na 700 etykietach. Patrzymy na **F1 makro** (nie accuracy), bo klasy rzadkie są tu najtrudniejsze – i to one pokazują, czy semi-supervised faktycznie pomógł.

### Oszczędność energii – uczciwie
- Względem **opisania wszystkich 100 000 zdjęć**: oszczędzamy przede wszystkim pracę ludzką – 99 000 zdjęć × ~10 s to około 275 roboczogodzin i odpowiadający im koszt.
- Ale uwaga: **self-supervised pretraining sam w sobie zużywa więcej energii niż zwykły fine-tuning** (dziesiątki godzin GPU). Jest tańszy niż trening dużej sieci od zera, ale nie jest "darmowy".
- Wniosek: główny zysk to koszt i czas etykietowania, nie kilowatogodziny. ROI liczymy na koszcie całkowitym (ludzie + sprzęt + energia), a nie na samym zużyciu prądu.

---

## Zadanie 15 – Reinforcement Learning Design

### Formalny opis problemu

**Agent:** system sterowania budynkiem. Kontroluje ogrzewanie (temperatura zadana), klimatyzację, wentylację, oświetlenie i rolety.

**Environment:** budynek wraz z pogodą i mieszkańcami.

**Stan (state):**
- temperatura wewnątrz (per strefa) i na zewnątrz,
- wilgotność, poziom CO2,
- obecność i liczba osób,
- pora dnia, dzień tygodnia, pora roku,
- prognoza pogody i nasłonecznienie,
- aktualna cena energii,
- stan urządzeń.

**Akcje (actions):**
- zmiana temperatury zadanej o ±0,5 °C,
- włączenie / wyłączenie / zmiana biegu wentylacji,
- ściemnienie lub wyłączenie oświetlenia,
- otwarcie lub zamknięcie rolet (darmowe ciepło i chłodzenie).

**Funkcja nagrody:**

```
r = - (a * zużyta_energia_kWh * cena)
    - (b * dyskomfort)
    - (c * liczba_przełączeń)

gdzie dyskomfort = odchylenie od strefy komfortu (20–24 °C),
liczone TYLKO gdy w pomieszczeniu są ludzie;
dodatkowa kara, gdy CO2 > 1000 ppm.
```

Kara za częste przełączanie chroni urządzenia przed zużyciem. Wagi a, b, c ustala właściciel budynku – to one definiują, ile komfortu jest wart kilowatogodzinę.

### Diagram agent-environment loop

```
        +-------------------+
        |      AGENT        |
        | (polityka sterow.)|
        +---+-----------+---+
            |           ^
   akcja a  |           |  stan s, nagroda r
   (grzej / |           |  (temperatura, CO2,
   przyciem)|           |   zużycie, obecność)
            v           |
        +---+-----------+---+
        |   ENVIRONMENT     |
        |  budynek + pogoda |
        |   + mieszkańcy    |
        +-------------------+
```

### Czy da się to rozwiązać supervised learning? Dlaczego RL jest lepsze

Supervised wymaga "poprawnych odpowiedzi". Tutaj ich nie ma – **nikt nie wie, jaka akcja jest optymalna**, znamy tylko skutki podjętych decyzji.

Supervised mógłby co najwyżej nauczyć się naśladować obecnego operatora, czyli **odtworzyć jego nieefektywność**. RL potrafi znaleźć strategię lepszą niż wszystko, co jest w danych historycznych.

Drugi, ważniejszy powód: **decyzje są sekwencyjne i mają skutki odroczone.** Podgrzanie budynku teraz wpływa na temperaturę za dwie godziny. Liczy się suma nagród w czasie, a nie trafność pojedynczej decyzji. Supervised uczy się punktowo, RL optymalizuje całą trajektorię.

**Uwaga praktyczna:** agenta uczymy najpierw w symulatorze (np. EnergyPlus). Eksploracja losowych akcji w prawdziwym budynku oznaczałaby zmarzniętych ludzi. Wdrażamy z twardymi limitami bezpieczeństwa, których agent nie może przekroczyć.

---

## Zadanie 16 – Multi-Model System Design

### Komponenty

| # | Zadanie | Typ ML | Podejście | Energochłonność |
|---|---|---|---|---|
| 1 | Klasyfikacja zdjęć produktów | Klasyfikacja obrazów | Supervised, transfer learning | Średnia |
| 2 | Generowanie opisów | Generowanie tekstu | Gotowy LLM + prompt / LoRA | **Wysoka** |
| 3 | Rekomendacje | Ranking | Supervised + collaborative filtering | Średnia, ale ogromna skala |
| 4 | Fałszywe recenzje | Klasyfikacja binarna + anomalie | Supervised + unsupervised | Niska |

### Architektura

```
     zdjęcie produktu
            |
            v
  [1] KLASYFIKATOR OBRAZÓW ---> kategoria + atrybuty
            |                            |
            v                            v
  embedding produktu  ----------> [2] GENERATOR OPISÓW
            |                            |
            |                            v
            |                       opis w bazie
            v                       (liczony RAZ)
  +---------------------+
  |    FEATURE STORE    | <--- historia zakupów, kliknięcia
  +----------+----------+
             |
             v
  [3] REKOMENDACJE  <--- filtr jakości <--- [4] DETEKTOR FAŁSZYWYCH RECENZJI
             |
             v
       lista dla użytkownika
```

### Jak komponenty współpracują
- Wspólny **feature store** i wspólne embeddingi produktów – liczone raz, używane przez wszystkie modele.
- Klasyfikator zdjęć nadaje kategorię, która jest wejściem dla generatora opisów i dla rekomendacji content-based (rozwiązuje zimny start nowych produktów).
- Detektor fałszywych recenzji obniża ocenę produktu, co wpływa na ranking – rekomendacje nie promują produktów z kupionymi opiniami.

### Najbardziej energochłonne
1. **Generator opisów (LLM)** – najdroższy na pojedyncze wywołanie.
2. **Rekomendacje** – pojedyncze zapytanie jest tanie, ale przy milionach zapytań dziennie sumarycznie mogą kosztować więcej niż LLM.

### Plan optymalizacji Green AI
1. **Nie trenować LLM od zera** – użyć gotowego modelu z dobrym promptem lub lekkim dostrojeniem (LoRA). To różnica rzędu tysięcy razy.
2. **Opisy generować raz i zapisywać w bazie.** Opis produktu nie zmienia się co sekundę – to nie musi być operacja czasu rzeczywistego. Największa pojedyncza oszczędność w całym systemie.
3. **Wspólny backbone wizyjny** dla klasyfikacji i wyszukiwania po obrazie zamiast dwóch osobnych sieci.
4. **Destylacja** dużego modelu do małego na potrzeby predykcji produkcyjnej.
5. **Dwuetapowe rekomendacje** (tani filtr → drogi ranking) + cache.
6. **Autoskalowanie GPU**, żeby karty nie stały włączone bez pracy – to częste i kosztowne marnotrawstwo.
7. **Monitoring kWh per komponent** – bez pomiaru nie ma optymalizacji.

---

## Zadanie 17 – Benchmark: Energy vs Accuracy (klasyfikacja tekstu)

Poniższe liczby to **szacunki rzędu wielkości** na typowym zbiorze kilkudziesięciu tysięcy dokumentów, nie wyniki pomiaru.

| Podejście | Accuracy | Energia treningu | Energia predykcji | Czas treningu | Czas predykcji | Łatwość wdrożenia |
|---|---|---|---|---|---|---|
| Keyword matching | 60–70% | ~0 (brak treningu) | ~0 | godziny pracy człowieka | < 1 ms | bardzo łatwe |
| Logistic Regression (TF-IDF) | 82–88% | ~0,01 kWh | ~0,001 Wh | minuty (CPU) | < 1 ms | łatwe |
| Random Forest | 83–88% | ~0,05 kWh | ~0,01 Wh | minuty–godziny (CPU) | 1–5 ms | łatwe |
| LSTM | 86–90% | ~1–5 kWh | ~0,1 Wh | godziny (GPU) | ~10 ms | średnie |
| BERT (fine-tuning) | 91–94% | ~5–20 kWh | ~0,5–1 Wh | godziny (GPU) | 20–50 ms (GPU) | trudniejsze |

**Ważne zastrzeżenie:** energia treningu BERT dotyczy tylko dostrojenia. Wstępne wytrenowanie modelu bazowego to rząd setek–tysięcy kWh, ale ponosi je raz autor modelu, a korzystają z niego wszyscy. To jest sedno idei transfer learningu jako praktyki Green AI.

### Wykres accuracy vs energia

```
accuracy
  93% |                                        BERT ●
  88% |              RF ●  ● LSTM
  85% |     LogReg ●
  65% | Keyword ●
      +--------------------------------------------------
        ~0      0,01     0,05     1-5        5-20   kWh
                        (skala logarytmiczna)
```

Skok z 65% na 86% kosztuje grosze. Skok z 86% na 93% kosztuje **sto do tysiąca razy więcej**.

### Rekomendacje

| Scenariusz | Wybór |
|---|---|
| Moderacja treści, miliony tekstów dziennie | Reguły + Logistic Regression |
| Analiza sentymentu w raportach, mały wolumen | BERT – jakość ma tu realną wartość |
| Urządzenie offline / telefon | Logistic Regression lub destylowany DistilBERT |
| Dane tabelaryczne z tekstem jako jedną z cech | Random Forest / gradient boosting |
| Pierwszy prototyp dowolnego projektu | **Zawsze Logistic Regression jako baseline** |

### Analiza Green AI efficiency
W przeliczeniu na accuracy na kilowatogodzinę **regresja logistyczna jest bezkonkurencyjna** – jest tańsza o kilka rzędów wielkości niż BERT, przy stracie 5–8 punktów procentowych.

Praktyczna zasada: **zacznij od najprostszego modelu, przechodź do droższego tylko wtedy, gdy różnica w jakości przekłada się na mierzalną wartość biznesową.** Bardzo często okazuje się, że nie przekłada się wcale – a baseline i tak jest potrzebny, żeby to w ogóle stwierdzić.

---

## Zadanie 18 – Ethical AI + Green AI (diagnostyka RTG)

### Aspekty etyczne: co jeśli model się myli?
Błędy nie są równe. **Fałszywie negatywny** (przeoczona choroba) jest znacznie groźniejszy niż fałszywie pozytywny (dodatkowe badanie). Próg decyzyjny ustawiamy więc tak, by maksymalizować czułość.

**Odpowiedzialność:** system musi być narzędziem wspierającym radiologa, a nie podejmującym decyzję. Odpowiedzialność za diagnozę zostaje przy lekarzu i placówce; producent odpowiada za jakość, walidację kliniczną i rzetelną dokumentację ograniczeń modelu. W UE taki system to wyrób medyczny i – zgodnie z AI Act – system wysokiego ryzyka: wymaga nadzoru człowieka, logowania decyzji i możliwości odwołania.

### Trade-off: czy 99% jest warte 100x energii wobec 97%?
**W tym przypadku tak – ale to zła alternatywa.**

Dwa punkty procentowe w diagnostyce to realni ludzie: przy 100 000 badań rocznie to 2 000 osób z inną diagnozą. Jednocześnie energia jednego badania AI jest znikoma wobec energii samego aparatu RTG i dojazdu pacjenta do szpitala.

Zamiast godzić się na stukrotny koszt, należy go najpierw zbić metodami, które nie kosztują jakości:
- **kaskada** – tani model odsiewa jednoznacznie prawidłowe zdjęcia, drogi ocenia resztę,
- **kwantyzacja i destylacja** modelu do predykcji,
- jedno porządne, dobrze zaplanowane trenowanie zamiast setek przypadkowych eksperymentów (to zwykle tu ucieka najwięcej energii).

### Fairness
Model trzeba oceniać **osobno dla każdej podgrupy**: płeć, wiek, pochodzenie, typ aparatu, szpital. Uśredniona dokładność potrafi ukryć, że dla jednej grupy model działa źle.

Klasyczna pułapka: model uczy się rozpoznawać **szpital** (po oznaczeniu na kliszy lub charakterystyce aparatu), a nie chorobę – bo w jednym szpitalu było więcej przypadków chorych. Zabezpieczenia: zróżnicowany zbiór treningowy, usuwanie oznaczeń ze zdjęć, obowiązkowa **walidacja zewnętrzna na danych z innej placówki**.

### Privacy
- Anonimizacja DICOM: usunięcie metadanych pacjenta oraz napisów wypalonych w obrazie.
- Świadoma zgoda na wykorzystanie danych do trenowania.
- Przetwarzanie lokalne (on-premise) zamiast wysyłania zdjęć do chmury.
- **Federated learning** – model jedzie do danych, dane nie opuszczają szpitala.
- Kontrola dostępu, logi, zgodność z RODO.

### Green AI
Transfer learning zamiast treningu od zera, ograniczenie liczby eksperymentów przez przemyślany plan, predykcja na CPU lub zoptymalizowanym akceleratorze, dostawca chmury z zieloną energią, pomiar i raportowanie zużycia.

### Framework decyzyjny: kiedy trade-off accuracy/energia jest akceptowalny
Cztery pytania w kolejności:

1. **Jaka jest cena błędu dla człowieka?** Zdrowie i życie → jakość wygrywa. Rekomendacja filmu → energia wygrywa.
2. **Czy różnica w jakości jest statystycznie istotna na niezależnym zbiorze testowym?** Jeśli 99% vs 97% mieści się w przedziale niepewności – wybierz tańszy model. Bardzo często tak właśnie jest.
3. **Czy tę jakość da się osiągnąć taniej inaczej?** Kaskada, destylacja, lepsze dane, lepsze etykiety. Zwykle da się.
4. **Jaki jest pełny bilans środowiskowy wobec alternatywy?** Dokładniejszy model może oszczędzić niepotrzebne powtórne badania i dojazdy – wtedy zużywa więcej prądu, a mniej zasobów ogółem.

**Reguła nadrzędna:** im wyższa stawka ludzka, tym mniej waży energia – ale zawsze najpierw wyczerp optymalizacje, które nic nie kosztują w jakości.

---

## Zadanie 19 – Data Science for Climate Change

### Wybrany problem
**Krótkoterminowa prognoza ryzyka pożaru lasu** – przewidywanie na 1–7 dni do przodu, w siatce 1 × 1 km. Służby dostają mapę ryzyka i mogą rozmieścić patrole oraz sprzęt tam, gdzie zagrożenie jest największe.

### Typ zadania ML i podejście
- **Klasyfikacja** (czy w danym kwadracie i dniu wybuchnie pożar) lub równoważnie **regresja** indeksu ryzyka.
- **Supervised.** Dane są tabelaryczne i przestrzenne, więc podstawą jest gradient boosting (XGBoost / LightGBM), opcjonalnie wzbogacony o cechy z obrazów satelitarnych.
- Wyzwanie: silnie niezbalansowane dane (pożary są rzadkie) – te same techniki co przy detekcji fraudu.

### Dane i źródła
| Dane | Źródło |
|---|---|
| Historia pożarów | NASA FIRMS, dane PSP / Lasów Państwowych |
| Pogoda i prognozy | ERA5 (Copernicus), IMGW |
| Wskaźniki roślinności (NDVI), wilgotność | Sentinel-2, Sentinel-3 |
| Typ lasu, wiek drzewostanu | Bank Danych o Lasach |
| Ukształtowanie terenu | numeryczny model terenu |
| Bliskość dróg i zabudowy | OpenStreetMap |

Prawie wszystkie źródła są otwarte i darmowe – to mocno poprawia ROI projektu.

### Pomiar impactu
- Odsetek pożarów, które wystąpiły w kwadratach oznaczonych jako wysokie ryzyko.
- Średni czas od zapłonu do podjęcia akcji gaśniczej.
- Spalona powierzchnia w hektarach, porównana rok do roku **i do regionu kontrolnego bez systemu**.
- Przeliczenie na CO2: 1 ha spalonego lasu to kilkadziesiąt ton uwolnionego CO2 plus utrata zdolności pochłaniania na lata. Oszczędzone tony = uniknięte hektary × współczynnik.

**Uczciwe zastrzeżenie:** atrybucja jest trudna – susza albo deszczowe lato wpłyną na wynik bardziej niż model. Dlatego konieczne jest porównanie z regionem kontrolnym, a nie samo "rok do roku".

### Green AI samego projektu
- Dane tabelaryczne + gradient boosting = godziny CPU, bez GPU.
- Predykcja raz dziennie dla całej siatki w jednym przebiegu wsadowym, nie na żądanie.
- Retrening raz na sezon.
- Buforowanie pobranych danych satelitarnych (pobieranie i dekodowanie to tu realny koszt).

Emisja projektu: rzędu pojedynczych ton CO2 rocznie, głównie z pracy zespołu.

### ROI
- **Koszt roczny:** dane darmowe, obliczenia kilkaset złotych miesięcznie, zespół 2 osób – główna pozycja.
- **Korzyść:** w regionie, gdzie rocznie płonie średnio 5 000 ha, redukcja spalonej powierzchni o 10% to 500 ha, czyli dziesiątki tysięcy ton niewyemitowanego CO2, plus uniknięte straty w drewnie i niższe koszty akcji gaśniczych.
- **Bilans:** emisja własna projektu to ułamek promila emisji unikniętej. ROI zdecydowanie dodatnie, nawet przy ostrożnych założeniach.

### Prezentacja dla stakeholderów (6 slajdów)
1. Problem i jego skala w liczbach.
2. Jakie dane mamy i skąd (podkreślić: darmowe).
3. Jak to działa – bez matematyki, jeden obrazek z mapą ryzyka.
4. Wyniki pilotażu na danych historycznych.
5. Koszt vs korzyść, w złotówkach i tonach CO2.
6. Plan wdrożenia, ryzyka i co jest potrzebne od odbiorcy.

---

## Zadanie 20 – Future of Green AI

> **Uwaga:** zgodnie z prośbą o zwięzłość poniżej jest skrócona wersja eseju (ok. 700 słów zamiast wymaganych w zadaniu 1000–1500). Struktura i bibliografia odpowiadają pełnej wersji – każdą sekcję można rozwinąć do wymaganej objętości.

### Jak będzie wyglądał Green AI w 2030 roku?

**Punkt wyjścia.** Przez ostatnią dekadę rozwój AI napędzała prosta zasada: większy model to lepszy model. Zapłaciliśmy za to rachunkiem energetycznym, który przestał być marginalny – centra danych odpowiadają już za kilka procent światowego zużycia energii elektrycznej, a prognozy mówią o dalszym szybkim wzroście napędzanym właśnie przez AI. Green AI to nurt, który pyta nie tylko "jak dokładny jest model", ale też "jakim kosztem".

**Trend 1: efektywność zamiast skali.** Najważniejsza zmiana do 2030 roku nie będzie technologiczna, tylko mentalna – efektywność stanie się parametrem jakości modelu, a nie dodatkiem. Widać to już dziś: małe, dobrze wytrenowane modele dorównują modelom kilkukrotnie większym sprzed dwóch lat. Techniki, które to umożliwiają – destylacja, kwantyzacja, przycinanie sieci, architektury mixture-of-experts aktywujące tylko potrzebny fragment modelu – przestaną być optymalizacją na końcu projektu, a staną się punktem wyjścia. Osobno rośnie znaczenie **inference**: model trenuje się raz, a odpowiada miliardy razy, więc to tam leży większość realnego zużycia energii.

**Trend 2: przesunięcie obliczeń na brzeg sieci.** Coraz więcej modeli będzie działać lokalnie – na telefonie, w samochodzie, na sterowniku przemysłowym. To oszczędza nie tylko energię transmisji, ale też chroni prywatność. Do 2030 roku uruchomienie przyzwoitego modelu językowego na laptopie będzie czymś całkowicie zwyczajnym.

**Trend 3: nowy sprzęt – z umiarkowanym optymizmem.** Układy neuromorficzne, naśladujące sposób działania neuronów, obiecują zużycie energii niższe o rzędy wielkości dla pewnych klas zadań i mają realną szansę wyjść z laboratoriów do zastosowań niszowych. Komputery kwantowe pozostaną natomiast w 2030 roku ciekawostką badawczą dla uczenia maszynowego – nie należy na nich opierać żadnych planów. Znacznie większy praktyczny wpływ będą miały zwykłe, kolejne generacje akceleratorów i lepsze chłodzenie centrów danych.

**Trend 4: regulacje.** To tu zmiana będzie najostrzejsza. Unijny AI Act już wprowadza obowiązki dokumentacyjne, a kierunek jest jasny: raportowanie śladu węglowego modeli stanie się obowiązkowe, podobnie jak dziś etykiety energetyczne sprzętu AGD. Do 2030 roku należy oczekiwać standardowej "karty energetycznej" modelu – ile kosztowało trenowanie, ile kosztuje jedna odpowiedź, w jakim regionie i z jakiego miksu energetycznego. Firmy, które nie mierzą tego dziś, będą miały problem.

**Napięcie: innowacja kontra środowisko.** To napięcie jest realne, ale często źle stawiane. AI zużywa energię, ale jednocześnie pomaga ją oszczędzać – w prognozowaniu produkcji z OZE, zarządzaniu sieciami energetycznymi, optymalizacji przemysłu i transportu. Właściwym pytaniem nie jest "czy AI zużywa dużo prądu", tylko "czy dany projekt oszczędza więcej, niż kosztuje". Ten bilans trzeba liczyć dla konkretnego zastosowania, bo czasem wychodzi świetnie, a czasem wcale.

**Rola Data Scientistów.** To my podejmujemy decyzje, które realnie przekładają się na zużycie energii – i robimy to codziennie, często bezrefleksyjnie. Wybór modelu, liczba eksperymentów, częstotliwość retreningu, region chmury, to czy wynik jest cache'owany. Żadna z tych decyzji nie wymaga zgody zarządu, a razem decydują o rachunku.

### Rekomendacje dla obecnych Data Scientists
1. **Mierz zużycie energii** swoich eksperymentów (np. CodeCarbon). Bez pomiaru nie ma optymalizacji.
2. **Zawsze zaczynaj od prostego baseline'u.** Często okaże się wystarczający – a bez niego nie wiesz, co zyskujesz na droższym modelu.
3. **Nie trenuj od zera, gdy istnieje gotowy model.** Transfer learning to najskuteczniejsza praktyka Green AI.
4. **Optymalizuj inference, nie tylko trening** – tam zwykle leży większość kosztu.
5. **Planuj eksperymenty.** Setka przypadkowych uruchomień kosztuje więcej niż dziesięć przemyślanych.
6. **Wybieraj region chmury z czystą energią** – ta sama praca, kilkukrotnie mniejsza emisja, zerowy koszt wdrożenia.
7. **Raportuj koszt energetyczny w dokumentacji modelu.** Niedługo będzie to obowiązek; lepiej mieć nawyk wcześniej.

### Bibliografia
1. Schwartz R., Dodge J., Smith N. A., Etzioni O. (2020), *Green AI*, Communications of the ACM, 63(12).
2. Strubell E., Ganesh A., McCallum A. (2019), *Energy and Policy Considerations for Deep Learning in NLP*, Proceedings of ACL.
3. Patterson D. et al. (2021), *Carbon Emissions and Large Neural Network Training*, arXiv:2104.10350.
4. Luccioni A. S., Viguier S., Ligozat A.-L. (2023), *Estimating the Carbon Footprint of BLOOM, a 176B Parameter Language Model*, Journal of Machine Learning Research.
5. International Energy Agency (2024), *Electricity 2024: Analysis and Forecast to 2026* – rozdział o zużyciu energii przez centra danych i AI.
6. Rozporządzenie Parlamentu Europejskiego i Rady (UE) 2024/1689 (*AI Act*).
