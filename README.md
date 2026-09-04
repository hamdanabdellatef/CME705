# CME705 Machine Learning

## Neural Networks, Deep Learning, and Research Practice

A 14-week graduate course developed for the Department of Computer Engineering at Karabuk University. The course connects machine-learning foundations with implementation and research practice. Students build core neural-network methods with NumPy, use PyTorch for modern deep-learning models, and develop a research question through a semester project.

This repository is the first reorganized release of material taught in Spring 2022. It establishes the complete course route, assessment structure, Python environment, tested example code, and research-project workflow. Detailed lecture notes and additional notebooks will be expanded in later releases.

## Start here

1. Read the [syllabus](syllabus.md) and [14-week schedule](schedule.md).
2. Follow [setup.md](setup.md) to create the Python environment.
3. Work through the linked module and lab for each week.
4. Begin the [research project](research-project/brief.md) in Week 1 and submit its milestones throughout the semester.

## Course outcomes

By the end of the course, students should be able to:

- formulate a machine-learning problem from an application need;
- design an evaluation that avoids data leakage and supports a defensible conclusion;
- implement and explain neural-network training with NumPy;
- train and inspect convolutional, sequence, and generative models with PyTorch;
- compare methods through controlled experiments and appropriate metrics;
- read research critically and turn experimental findings into a feasible next research question; and
- release code and results that another researcher can reproduce.

## Repository map

- [`modules/`](modules/README.md): topic guides and learning outcomes
- [`labs/`](labs/README.md): runnable NumPy and PyTorch examples
- [`assignments/`](assignments/README.md): semester assignments
- [`research-project/`](research-project/brief.md): project brief, milestones, rubric, and templates
- [`readings/`](readings/README.md): annotated links to official and primary sources
- [`data/`](data/README.md): dataset and split policy
- [`instructor-notes/`](instructor-notes/README.md): delivery and assessment guidance

## Technology

The course uses Python, NumPy, and PyTorch. MATLAB material from the historical course is intentionally excluded.

## Release status

Version 0.1.0 is the first public-ready structure. The examples use generated data so the repository remains small and its basic checks can run without downloading datasets. Dataset-based notebooks and complete lecture notes are planned as later, separately reviewed additions.

## Citation and license

Cite the course using [`CITATION.cff`](CITATION.cff). Course text is licensed under CC BY 4.0 and original code under the MIT License; see [`LICENSE.md`](LICENSE.md). Third-party books, slides, source trees, student records, and student submissions are not included.
