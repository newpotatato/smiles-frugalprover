# FrugalProver — план на команду (3 человека)

Обновлено: 2026-07-12. Прувер/решение задач НЕ трогаем — маленькая модель нужна
только чтобы снять активации gℓ(x); генерация решений/бюджет — три отдельных,
управляемых по стоимости режима (см. §0).

Это план ПО ЛЮДЯМ/времени. Научное содержание каждого направления (вопрос,
метод, что считать полезным результатом) — в [RESEARCH_PLAN.md](RESEARCH_PLAN.md).

## 0. Три уровня данных (см. `data_prep.py`, `colab_budget_oracle_pilot.ipynb`)

| Тир | Что делает | Стоимость | Файл-источник | Выход |
|---|---|---|---|---|
| **activations_only** | phi(x) + gℓ(x) на **всех слоях**. Никакой генерации. | ~бесплатно, масштабируется на сотни задач | `math_pool_large_partN.json` (300 задач всего) | `pilot_activations_only_partN.jsonl` |
| **full_sweep** | Полный перебор бюджетов × N_SAMPLES, честный B*/p(B) **с доверительным интервалом Уилсона** (адекватная оценка, не просто точка) | дорого, поэтому — на меньшей подвыборке | `math_labeling_subset_partN.json` (80 задач, вложены в large_pool по id) | `pilot_full_sweep_partN.jsonl` |
| **baseline_single** | ОДИН бюджет × несколько сэмплов — грубый, но дешёвый сигнал решаемости на большем масштабе, чем full_sweep | дёшево-средне | `math_pool_large_partN.json` (можно взять кусок) | `pilot_baseline_single_partN.jsonl` |

`math_labeling_subset.json`'ы id — подмножество `math_pool_large.json`, поэтому `merge_results.py` **джойнит** записи по id: одна и та же задача может получить активации из тира 1 и B*/p(B) из тира 2 — не задваивая работу.

| Файл | Что делает | Статус |
|---|---|---|
| [data_prep.py](data_prep.py) | Строит large pool (300, сбалансированный) + вложенную labeling subset (80) | ✅ прогнан: 300 (53-67/уровень), 80 (14-18/уровень) |
| [split_subset.py](split_subset.py) | Универсальный сплиттер: `python split_subset.py <файл> <N>` — режет любой файл на N частей | ✅ прогнан на обоих файлах |
| [colab_budget_oracle_pilot.ipynb](colab_budget_oracle_pilot.ipynb) | `RUN_MODE` = activations_only / full_sweep / baseline_single, все слои, Wilson CI | Готов к загрузке |
| [merge_results.py](merge_results.py) | ID-based join across тиров (не конкатенация!) | ✅ прогнан на 3 фейковых тирах — джойн корректный |
| [analyze.py](analyze.py) | Полный layer-sweep + confirmatory/exploratory (Bonferroni) + accuracy@budget — работает на смеси тиров, использует нужный тир для каждой секции | ✅ прогнан на 200-record синтетике (60 full_sweep, 30 baseline, 110 только активации) |
| [oracle.py](oracle.py) | `--mode regression/classification`, `--compare-models` (Ridge/ElasticNet/MLP/GP или Logistic/MLP/GP) | ✅ прогнан на той же синтетике |
| [id_vs_difficulty.py](id_vs_difficulty.py) | **3 метода сжатия**: TwoNN, MLE (Levina-Bickel), participation ratio — согласны ли нелинейные методы (TwoNN/MLE) друг с другом, и линейный (PR) с ними | ✅ прогнан — использует все 200 записей для геометрии, только 60 full_sweep для solvability-корреляции |
| [calibration_cost.py](calibration_cost.py) | Три числа: цена калибровки (только на labeled-тире) vs oracle-guided на том же n vs **амортизация на весь large pool** | ✅ прогнан: калибровка 60 задач → амортизация на 200 |
| [learning_curve.py](learning_curve.py) | "updated online as outcomes accrue" — растёт ли AUC с ростом n (только на full_sweep-тире) | ✅ прогнан |
| [twonn.py](twonn.py) | TwoNN + **MLE (Levina-Bickel)** — два независимых нелинейных ID-эстиматора | ✅ оба сверены на синтетике (истинная размерность восстанавливается) |

## 1. Роли — три независимые задачи

- **A — Oracle experiments.** `oracle.py --mode classification --compare-models --demo`, `calibration_cost.py` (амортизация на весь пул), при желании `learning_curve.py`.
- **B — Layer empirics.** `analyze.py` — полный sweep по всем слоям, confirmatory (a priori, последний слой) vs exploratory (argmax, с поправкой Бонферрони).
- **C — ID/сложность гипотеза.** `id_vs_difficulty.py` — 3 метода сжатия (TwoNN, MLE, participation ratio) по уровням MATH + per-problem локальная плотность.

Все трое сначала параллельно прогоняют **activations_only** на своей трети `math_pool_large_partN.json` (это основной сегодняшний результат — дёшево, быстро, масштабно). full_sweep/baseline_single на `math_labeling_subset_partN.json` — следующий шаг, когда решите, сколько бюджета на это выделить (не обязательно сегодня).

## 2. Что проверить перед реальным прогоном

- Dry-run на N_PROBLEMS=3 в каждом RUN_MODE перед полным прогоном — так делали раньше, работает так же.
- `RUN_MODE`/`RUN_TAG`/`SUBSET_FILE` — три переменные вверху ноутбука, меняются перед каждым прогоном.
- Для activations_only можно смело поднимать `N_PROBLEMS` до всей `math_pool_large_partN.json` (100 на человека) — стоимость чисто forward-pass.

## 3. Вопросы комиссии — новое

- *"Почему часть задач размечена по бюджету, а часть — только активации?"* → Сознательное разделение по стоимости: активации дёшевы и масштабируются, полная разметка B* дорога (сама разметка тратит токены — см. `calibration_cost.py`). Вложенность гарантирует, что размеченные задачи — не отдельная выборка, а подмножество большого пула.
- *"Адекватна ли оценка B* при N_SAMPLES~3?"* → Даём доверительный интервал Уилсона на p(B), а не голую точку — честно показывает, сколько шума в каждой B*-метке.
- *"Один метод сжатия/ID — мало доверия. Проверяли другие?"* → Да: TwoNN и MLE (Levina-Bickel) — два независимых нелинейных эстиматора; PCA participation ratio — линейный. Если TwoNN и MLE расходятся по направлению — предупреждаем прямо в выводе скрипта, не прячем.
- *"Экономика оракула — только на размеченных задачах?"* → `calibration_cost.py` считает и амортизацию: обученный на labeled-подвыборке оракул применяется бесплатно (по активациям) ко всему large pool — это и есть реальный payoff масштаба.
