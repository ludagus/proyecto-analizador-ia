DROP TABLE IF EXISTS analysis;

CREATE TABLE analysis (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    code TEXT,
    lines INT,
    complexity INT,
    time_taken REAL,
    errors INT,
    prediction TEXT,
    feedback TEXT
);

INSERT INTO analysis (code, lines, complexity, time_taken, errors, prediction, feedback)
VALUES ("print(""Hola mundo"")", 4, 1, 0.21, 0, "Bajo", "Buen dominio. Seguimiento estándar recomendado.");
INSERT INTO analysis (code, lines, complexity, time_taken, errors, prediction, feedback)
VALUES ("for i in range(5): print(i)", 9, 2, 0.53, 1, "Medio", "Se detectan algunas dificultades. Reforzar conceptos.");
INSERT INTO analysis (code, lines, complexity, time_taken, errors, prediction, feedback)
VALUES ("def f():
 for i in range(10):
  if i % 2 == 0: print(i)", 14, 5, 1.43, 0, "Alto", "Dificultades significativas. Intervención personalizada necesaria.");
INSERT INTO analysis (code, lines, complexity, time_taken, errors, prediction, feedback)
VALUES ("while True:
 break", 7, 3, 0.74, 0, "Medio", "Se detectan algunas dificultades. Reforzar conceptos.");
INSERT INTO analysis (code, lines, complexity, time_taken, errors, prediction, feedback)
VALUES ("x = 2 + 2", 3, 0, 0.12, 0, "Bajo", "Buen dominio. Seguimiento estándar recomendado.");
