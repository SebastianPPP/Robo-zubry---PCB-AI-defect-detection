# Detekcja defektów 
## Założenia projektowe:
Skorzystanie z gotowego, dostępnego datasetu zawierającego zdjęcia z defektami na płytkach PCB.
Uzasadnij wybor dobranego datasetu.
Odpowiednio przygotuj dane, znormalizuj (np. torch.transform).
Dobór i przygotowanie modelu - zakładamy użycie biblioteki pytorch.
Przygotowanie aplikacji GUI do prezentacji wyników.
Przygotowanie prezentacji opisującej proces.


[Link do datasetu:](https://www.kaggle.com/datasets/norbertelter/pcb-defect-dataset)


## Dataset
For our project we chose kaggle dataset from here:
[Link](https://www.kaggle.com/datasets/norbertelter/pcb-defect-dataset)
It contains 10668 images and the corresponding annotation files, which is perfect for this project.
Classified defects:
- missing hole,
- mouse bite,
- open circuit,
- short, 
- spur,
- spurious copper.

The data has enough images to train our model to detect defects, and if we are not focused on the percentage maxing it should be good choice. The quality of dataset is great and the resolution satisfies our needs.