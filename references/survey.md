<!-- ===== 01_Ansuini_IntrinsicDimension_NeurIPS2019.pdf ===== -->

# Intrinsic dimension of data representations in deep
neural networks

# Alessio Ansuini

# Alessandro Laio

International School for Advanced Studies
alessioansuini@gmail.com

International School for Advanced Studies
laio@sissa.it

# Jakob H. Macke

# Davide Zoccolan

Technical University of Munich
macke@tum.de

International School for Advanced Studies
zoccolan@sissa.it

# Abstract

Deep neural networks progressively transform their inputs across multiple pro-
cessing layers. What are the geometrical properties of the representations learned
by these networks?
Here we study the intrinsic dimensionality (ID) of data-
representations, i.e. the minimal number of parameters needed to describe a repre-
sentation. We find that, in a trained network, the ID is orders of magnitude smaller
than the number of units in each layer. Across layers, the ID first increases and
then progressively decreases in the final layers. Remarkably, the ID of the last hid-
den layer predicts classification accuracy on the test set. These results can neither
be found by linear dimensionality estimates (e.g., with principal component anal-
ysis), nor in representations that had been artificially linearized. They are neither
found in untrained networks, nor in networks that are trained on randomized labels.
This suggests that neural networks that can generalize are those that transform the
data into low-dimensional, but not necessarily flat manifolds.

# 1
Introduction

Deep neural networks (DNNs), including convolutional neural networks (CNNs) for image data,
are among the most powerful tools for supervised data classification. In DNNs, inputs are se-
quentially processed across multiple layers, each performing a nonlinear transformation from a...

*[figure -- see caption above/below]*

*Figure 1: The TwoNN estimator derives an estimate of
intrinsic dimensionality from the statistics of nearest-
neighbour distances.*

# 2
Estimating the intrinsic dimension of data representations

Inferring the intrinsic dimension of high-dimensional and sparsely sampled data representations is
a challenging statistical problem. To estimate the ID of data-representations in deep networks, we
leverage a recently developed global ID-estimator (‘TwoNN’) that is based on compu...

# 3
Results

## 3.1
The intrinsic dimension exhibits a characteristic shape across several networks

Our first goal was to empirically characterize the ID of data representations in different layers of
deep neural networks. Given a layer l of a DNN, an individual data point (e.g., an image) is mapped
onto the set of activations of all the nl units of the layer, which define a po...

*[figure -- see caption above/below]*

*Figure 2: Modulation of ID across hidden layers of deep convolutional networks A) ID across
layers of VGG-16-R, error bars are the standard deviation of the ID (see A.1). Numbers in plot in-
dicate embedding dimensionality of each layer. B Subsampling analysis on VGG-16-R experiment,
reported for the same layers as in the inset in A (see A.1 for details).*

*Figure 3: ID of object manifolds across networks. A) IDs of data representations for 4 networks:
each point is the average of the IDs of 7 object manifolds. The error bars are the standard deviations
of the ID across the single object’s estimates (see A.1). B) The ID as a function of the relative depth
in 14 deep convolutional networks spanning different sizes, architectures and training techniques.
Despite the wide diversity of these models, the ID profile follows a typical hunchback shape (error
bars not shown).*

## 3.2
The intrinsic dimension of the last hidden layer predicts classification performance

Although the hunchback shape was preserved across networks, the IDs in the last hidden layers
were not exactly the same for all the networks. To better resolve such differences, we computed the
ID in the last hidden layer of each network using a much larger pool of images of the ...

## 3.3
Data representations lie on curved manifolds

The strength of the TwoNN method lies in its
ability to infer the ID of data representations,
even if they lie on curved manifolds. This raises
the question of whether our observations (low
IDs, hunchback shapes, correlation with test-
error) reflect the fact that data points liv...

*[figure -- see caption above/below]*

*Figure 4: ID of the last hidden layer predicts
performance.
The ID of data representations
(training set) predicts the top 5-score performance
on the test set. Inset Detail for the ResNet class.*

*Figure 5: Evidence that data-representations are on curved manifolds A) Variance spectra of last
hidden layer do not show a clear gap. B) ID in the last hidden layer of VGG-16 (black), compared
with the ID of a synthetic Gaussian dataset with the same size and second-ordercorrelations structure
(red). C) The ID and the PC-ID along the layers of VGG-16 for a trained network and an untrained,
randomly initialized network. The ED, rescaled to reach the maximum at 400, is shown in blue.*

## 3.4
The initial increase in intrinsic dimension can arise from irrelevant features

We generally found the ID to increase in the initial layers. However, this was not observed for a
small network trained on the MNIST data-set (Fig. 6B, black curve) and was also less pronounced
for AlexNet (Fig. 3A, orange curve). A mechanism underlying the initial ID rise could ...

*[figure -- see caption above/below]*

*Figure 6: A The addition of a luminance gradient across the images of the MNIST dataset results
in a stretching of the image manifold along a straight line in the input space of the pixel representa-
tion. B Change of the ID along all the layers of the MNIST network, as obtained in three different
experiments: 1) with the original MNIST dataset (black curve) 2) with the luminance-perturbed
MNIST⋆dataset (blue curve) and 3) with the MNIST†, in which the label of the MNIST images
where randomly shuffled (red curve).*

## 3.5
A network trained on random labels does not show the characteristic hunchback profile
of ID variation

In untrained networks the ID profile is largely flat (Fig. 5C). Are there other circumstances in which
the ID profile deviates from the typical hunchback shape of Figs 2A and 3A,B, with IDs that do not
decrease progressively towards the output? It turns out that this is the case ...

# 4
Conclusions and Discussion

Convolutional neural networks, as well as their biological counterparts, such as the visual system
of primates [25] and other species [24, 26], transform the input images across a progression of
processing stages, eventually providing an explicit (i.e. transformation-tolerant) re...

*[figure -- see caption above/below]*

*Figure 7: A. Input layer. The intrinsic dimen-
sionality of the data can assume low values
due to the presence of irrelevant features un-
correlated with the ground truth. B. The first
hidden layers pre-process the data raising its
intrinsic dimension. C,D. The representation
is squeezed onto manifolds of progressively
lower intrinsic dimension. These manifolds
are typically not hyperplanes. D. In the last
hidden layer (D) the ID shows a remarkable
correlation with the performance in trained
networks. E. The output layer.*

# Acknowledgments

We thank Eis Annavini for providing the custom dataset described in A.1.1; Artur Speiser and Jan-
Matthis Lückmann for a careful proofreading of the manuscript. We warmly thank Elena Facco
for her valuable help in the early phases of this project. We also thank Naftali Tishby, Ri...

# References

# A
Appendix

## A.1
Details of numerical experiments

All our experiments were performed in PyTorch [35] (version 1.0) on a Linux workstation
with 64GB of RAM and a GeForce GTX 1080 Ti NVIDIA graphic card.
The code to com-
pute the ID estimates with the TwoNN method and to reproduce our experiments is available
at this repository: g...

### A.1.1
Datasets

Custom dataset
A dataset of 1400 images developed for a neurophysiological study [20]. The
dataset consisted of 40 three-dimensional (3D), computer graphics models of both natural and man-
made objects, each rendered in 36 different views, obtained by combining in-plane and in-de...

### A.1.2
Architectures

We describe the architectures used in order of appearance in the main text.

VGG-16-R
We removed the last hidden layers (the last convolutional and all the dense layers) of
a VGG-16 network [19] pre-trained on ImageNet [11] and substituted it with randomly initialized
layers of the...

### A.1.3
Estimating intrinsic dimension

Experiments with the custom dataset
In this experiment (Fig. 2A,B), we fine-tuned the last
layers of a VGG-16 network pre-trained on ImageNet using the ≈80% of the 1440 images of the
dataset in [20] (30 images for each category in the training set, the remaining 6 images for each...

### A.1.4
Further results

Variability of the ID across object categories
In sec-
tion 3.1, we showed that, across layers, the ID displays a
typical ‘hunchback’ profile. In particular, the average of
seven class-specific ID profiles is shown in Fig. 3A,B and
their standard deviation is reported in Fig. 3A....

*[figure -- see caption above/below]*

*Figure 8: The ID variation across lay-
ers is generally consistent across object
classes. This figure refers to AlexNet in
Fig. 3, where the average across classes
is reported.*

*Figure 9: Dynamics of the ID on a VGG-16 trained on CIFAR-10. A Dynamics of the ‘hunchback’
shape for a typical training history. The black thick line refers to the untrained network. The color
codes for the training epochs. B Training and test accuracy (dashed red and blue curves respectively)
and ID in the last hidden layer in the early phases. The dynamics of the ID is non-monotonic; at the
same time there is no substantial overfitting. C ID vs. test error (same data as in B).*

---

<!-- ===== 02_Facco_TwoNN_SciReports2017.pdf ===== -->

# (untitled section)

# Estimating the intrinsic dimension
of datasets by a minimal 
neighborhood information

OPEN

Received: 7 June 2017

Elena Facco  , Maria d’Errico, Alex Rodriguez & Alessandro Laio

23 August 2017

Accepted:

Analyzing large volumes of high-dimensional data is an issue of fundamental importance in data 
science, molecular simulations and beyond. Several approaches work on the...

# Results

Let i be a point in the dataset, and consider the list of its first k nearest neighbors; let r1, r2, …, rk be a sorted list of 
their distances from i. Thus, r1 is the distance between i and its nearest neighbor, r2 is the distance with its second 
nearest neighbor and so on; in ...

*[figure -- see caption above/below]*

*[figure -- see caption above/below]*

*[figure -- see caption above/below]*

*[figure -- see caption above/below]*

*[figure -- see caption above/below]*

*[figure -- see caption above/below]*

*Figure 1. The fitting function l(x) in three exemplar datasets of 2500 points. In the first column we display 
the dataset while in the second one we represent dataset S (red dots) together with the discarded points (gray 
dots) and the fitting function l(x). Panel A, A’: cube in dimension 14 (in panel A only the first 3 coordinates are 
represented) analyzed with pbc. Panel B, B’: a Swiss Roll. Panel C, C’: a Cauchy dataset in dimension 20 (only 
the first 3 coordinates are represented).*

# (untitled section)

# (untitled section)

*[figure -- see caption above/below]*

*[figure -- see caption above/below]*

# (untitled section)

*[figure -- see caption above/below]*

*[figure -- see caption above/below]*

*Figure 2. Scaling of the estimated ID with respect to the number of points; for each distribution and for a 
number of points going from 20 to 25000 we harvest 200 instances of the dataset and average the resulting 
estimates for the ID. The test is carried out in dimension 2, 5 and 10. Panel A: Hypercube with pbc. Panel B:
gaussian distribution. Panel C: Cauchy dataset. Panel D: uniform distribution on a hypersphere.*

to changes in the neighborhood size like in a standard block analysis. In the case of TWO-NN it is possible to 
modify the neighborhood size by reducing the number of points in the dataset: the smaller N, the larger the aver-
age distance to the second neighbor. In practice, simi...

*[figure -- see caption above/below]*

*Figure 3. Estimated dimension d vs the number of points N in logarithmic scale; for each value of N the dataset
is partitioned in a number of independent sets containing exactly N points, d is computed on each subdataset 
and a measure d(N) is obtained as an average of these values. In Panel A we study the case of a uniform plane of 
50000 points in dimension 2 perturbed by a Gaussian noise with variance σ along 20 independent directions; 
σ takes the three values 0.0, 0.0001 and 0.0002. In Panel B we analyze a dataset composed of a two-dimensional 
Gaussian of 50000 points wrapped around a Swiss Roll and perturbed by a gaussian noise with variance σ along 
20 independent directions. Again σ takes the three values 0.0, 0.0001 and 0.0002.*

*Figure 4. Scaling of the estimated ID with respect to the number of points for ISOMAP face (panel A) and
MNIST database (panel B).*

*Figure 5. Scaling of the estimated ID with respect to the number of configurations for the dynamics of 
trinucleotide AAA in case of RMSD distances (panel A) and TICA distances (panel B). On the left a possible 
configuration is represented.*

# Discussion

Discussion
In this work we address the problem of finding the minimal number of variables needed to describe the relevant 
features of a dataset; this number is known as intrinsic dimension (ID). We develop TWO-NN, an ID estimator that 
employs only the distances to the first two...

# References

# Acknowledgements
We want to acknowledge

We want to acknowledge Daniele Granata and Alex Rodriguez for their useful advice. We also acknowledge
Michele Allegra, Giovanni Pinamonti and Antonietta Mira.

# Author Contributions

Laio, Facco, d’Errico and Rodriguez designed and performed the research. Laio and Facco wrote the manuscript
text, prepared the figures and reviewed the manuscript.

# Additional Information
Supplementary information a

Supplementary information accompanies this paper at https://doi.org/10.1038/s41598-017-11873-y.

Competing Interests: The authors declare that they have no competing interests.

Publisher's note: Springer Nature remains neutral with regard to jurisdictional claims in published maps a...

---

<!-- ===== 03_Hendrycks_MATH_NeurIPS2021.pdf ===== -->

# Measuring Mathematical Problem Solving With the
MATH Dataset

# Abstract

Many intellectual endeavors require mathematical problem solving, but this skill
remains beyond the capabilities of computers. To measure this ability in machine
learning models, we introduce MATH, a new dataset of 12,500 challenging
competition mathematics problems. Each problem in MATH has a full step-by-step
solution which can be used to teach models to generate answer derivations and
explanations. To facilitate future research and increase accuracy on MATH, we
also contribute a large auxiliary pretraining dataset which helps teach models the
fundamentals of mathematics. Even though we are able to increase accuracy on
MATH, our results show that accuracy remains relatively low, even with enormous
Transformer models. Moreover, we find that simply increasing budgets and model
parameter counts will be impractical for achieving strong mathematical reasoning
if scaling trends continue. While scaling Transformers is automatically solving
most other text-based tasks, scaling is not currently solving MATH. To have more
traction on mathematical problem solving we will likely need new algorithmic
advancements from the broader research community.

# 1
Introduction

Mathematics is a highly effective tool in many intellectual endeavors. It enables us to count and
quantify objects, and it can be relied upon because it is consistent and based on logic. Mathematics
pervades the sciences and can be used to model planetary orbits, atomic motion, s...

# Metamath Theorem Proving

# MATH Dataset (Ours)

Problem:
Tom has a red marble, a green marble,
a blue marble, and three identical yellow marbles.
How many different groups of two marbles can
Tom choose?

n ∈N ∧n+1
2
∈N =⇒∃m ∈N : n = 2m + 1.
GPT-f’s generated proof:

|- ((N e. NN0 /\ ((N + 1)/2) e.
NN0) -> ((N - 1) / 2) e. NN0)
|- ...

# DeepMind Mathematics Dataset

Divide 1136975704 by -142121963.
A: -8
Let k(u) = u**2+u-4. Find k(0).
A: -4
Sort 2, 4, 0, 6.
A: 0, 2, 4, 6
Solve 4 - 4 - 4 = 188*m for m.
A: -1/47

Solution:
Complete the square by adding 1 to
each side. Then (x + 1)2 = 1 + i = e
iπ
4 √
2, so
x + 1 = ±e
iπ
8
4√
2. The desired prod...

# 2
Related Work

Neural Theorem Provers.
Much of the existing work on machine learning models for mathemati-
cal reasoning relies on automated theorem proving benchmarks. Huang et al. (2019) use the Coq
theorem proving environment to create a machine learning benchmark with 1, 602 theorems and
le...

*[figure -- see caption above/below]*

*Figure 2: Compared to existing proof and plug-
and-chug tasks, our mathematical problem solving
task is considerably more challenging. HOList re-
sults are from Wu et al. (2021). HOLStep results
are from Crouse et al. (2019). DeepMind Math ac-
curacy is the median IID accuracy from Henighan
et al. (2020). Symbolic Integration accuracy is
from Lample and Charton (2020).*

# 3
Datasets

In this section, we introduce two new datasets, one for benchmarking mathematical problem-solving
ability (MATH) and one for pretraining (AMPS).

## 3.1
The MATH Dataset

The MATH dataset consists of problems from mathematics competitions including the
AMC 10, AMC 12, AIME, and more.
Many of these problems can be collected from
aops.com/community/c3158_usa_contests. These competitions span decades and assess the mathe-
matical problem-solving abil...

*[table -- see caption above/below]*

*Table 2: MATH accuracies across subjects. ‘*’ indicates that the model is a few-shot model. The
character ‘B’ denotes the number of parameters in billions. The gray text indicates the relative
improvement over the 0.1B baseline. All GPT-2 models pretrain on AMPS, and all values are
percentages. GPT-3 models do not pretrain on AMPS due to API limits. Model accuracy is increasing
very slowly, so much future research is needed.*

# Human-Level Performance.

Human-Level Performance.
To provide a rough but informative comparison to human-level
performance, we randomly sampled 20 problems from the MATH test set and gave them to humans.
We artificially require that the participants have 1 hour to work on the problems and must perform
ca...

## 3.2
AMPS (Khan + Mathematica) Dataset

Since pretraining data can greatly influence performance (Hernandez et al., 2021; Gururangan et al.,
2020) and since mathematics is a small fraction of online text, we introduce a large and diverse
mathematics pretraining corpus. Our pretraining dataset, the Auxiliary Mathematics...

# 4
Experiments

In this section, we perform experiments to investigate performance on the MATH dataset. We find
that accuracy remains low even for the best models. Furthermore, unlike for most other text-based
datasets, we find that accuracy is increasing very slowly with model size. If trends c...

## 4.1
Experimental Setup

Models and Hyperparameters.
Because MATH answers must be generated, we use autoregressive
language models, namely GPT-2 (Radford et al., 2016) and GPT-3 (Brown et al., 2020), which are
decoder models pretrained on natural language text. Our GPT-2 models tokenizes numbers so that
...

*[figure -- see caption above/below]*

*[figure -- see caption above/below]*

*[figure -- see caption above/below]*

## 4.2
Analyzing Model Performance

AMPS Pretraining.
As an ablation, we test how models with AMPS pretraining compare with
models that were not pretrained on AMPS. Without pretraining on AMPS, a GPT-3 (13B) model
fine-tuned on MATH attains 5.2% accuracy. In contrast, a GPT-2 (0.1B) model both pretrained on
AMPS an...

*[figure -- see caption above/below]*

## 4.3
Analyzing Step-by-Step Solutions

Scratch Space.
Our MATH dataset and AMPS pretraining dataset provide full step-by-step solu-
tions, an important and rare type of side information (Murty et al., 2020) that can in principle teach
models how to derive answers and use scratch space. By training a language model on ...

# MATH Accuracy Given Partial Solutions

*[figure -- see caption above/below]*

*Figure 5: Models conditioned on most of a problem’s
step-by-step solution can often understand the solution to
predict the final answer. Not all solutions have an answer
that is immediate from the preceding solution text, though
many are. ‘99%’ of a solution is all the solution text be-
fore the final answer. This demonstrates that, even with
substantial help, models are still struggling.*

Examples.
We can also qualitatively
assess the step-by-step solutions that the
model generates. We show examples of
generated solutions in Figures 3 and 4.
We find that the model can consistently
generate correct LATEX and often per-
forms steps that appear related to the
questio...

# 5
Conclusion

In this paper, we laid groundwork for future research in machine learning for mathematical problem
solving. We introduced the MATH benchmark, which enables the community to measure mathe-
matical problem-solving ability. In addition to having answers, all MATH problems also inclu...

# References

# A
Appendix

In this appendix, we have more comparisons with previous datasets, a discussion of logic and
intelligence tests, further AMPS and MATH details, an analysis of model performance as difficulty
level changes, and results with the BART architecture.

## A.1
Expanded Dataset Comparisons

We compared to ten datasets in the main paper, and now we will further describe plug-and-chug
datasets. Dolphin18K (Huang et al., 2016) is one of the first modern datasets in this space and is based
on Yahoo! Answers and includes questions such as “help!!!!!!!(please) i cant figu...

## A.2
Logic and Intelligence Tests

*Model Size vs. LogiQA Accuracy*

*[figure -- see caption above/below]*

*Figure 6: Difficult natural language tasks such as LogiQA will soon be solved just by making models
larger, assuming trends continue. The Transformers in this figure are UnifiedQA (Khashabi et al.,
2020) models of various sizes.*

While enormous Transformers perform poorly on MATH, they do well on other logic and intelligence
tests.

We analyze Transformers on LogiQA (Liu et al., 2020), a task with logical reasoning questions such
as “David knows Mr. Zhang’s friend Jack, and Jack knows David’s friend Ms. Lin...

*Figure 7: Example of asymptote code and the figure it produces.*

## A.3
Further Dataset Information

Rendering Graphics.
For the first time, our dataset makes it possible for text-based models to
process graphical mathematical figures by expressing figures in asymptote code. For example,
Figure 7 shows asymptote code and the figure it produces. In short, it is possible to concis...

## A.4
Difficulty Analysis

We break down MATH accuracy by difficulty levels. In Figure 9, we observe that human difficulty
and machine difficulty track each other. In Figure 10, we find that accuracy can vary by level and
subject substantially. Finally, in Figure 11a and Figure 11b, we analyze the relation...

## A.5
Results with the BART Architecture

We use BART (Lewis et al., 2020) to determine whether other existing architectures can improve
performance. In the main paper we analyzed the performance of various GPT models, which are
unidirectional decoder models. Lewis et al. (2020) introduce BART, which has a bidirectional ...

*Figure 8: A Khan Academy problem and solution, followed by the code for a simple Mathematica
module used to generate polynomials GCD problems. These problems are available in AMPS.*

## A.6
Further Human Evaluation Details

Because MATH requires a strong mathematical background to perform well on, and a long amount
of time to solve problems, we were restricted to assessing six human participants and could not
rely on crowdsourcing sites such as Amazon Mechanical Turk. All participants are university...

*[figure -- see caption above/below]*

*Figure 9: Problems that are more difficult for humans are also more difficult for GPT-2.*

### MATH Accuracies Across Subject Difficulty Levels

*[figure -- see caption above/below]*

*Figure 10: Accuracy per subject per difficulty level.*

*(b) Subject accuracy vs solution length. Each point
represents a subject at a specific difficulty level. We
exclude problems with asymptote figures. Results are
from GPT-2 (1.5B).*

where −1 ≤x ≤1?

# B
Checklist Information

Legal Compliance.
We create and collect various mathematics problems to create MATH and
AMPS.

AMPS consists of problems generated with Mathematica and Khan Academy code. Mathematica
serves as a calculator and does not copyright its numerical answer outputs, in much the same way
th...

*Figure 13: Khan Academy modules in AMPS (Part 2).*

*Figure 14: Khan Academy modules in AMPS (Part 3).*

*Figure 15: Khan Academy modules in AMPS (Part 4).*

---

<!-- ===== 04_Lightman_LetsVerifyStepByStep_ICLR2024.pdf ===== -->

# Let’s Verify Step by Step

# Abstract

In recent years, large language models have greatly improved in their
ability to perform complex multi-step reasoning.
However, even state-
of-the-art models still regularly produce logical mistakes. To train more
reliable models, we can turn either to outcome supervision, which provides
feedback for a final result, or process supervision, which provides feedback
for each intermediate reasoning step. Given the importance of training
reliable models, and given the high cost of human feedback, it is impor-
tant to carefully compare the both methods. Recent work has already
begun this comparison, but many questions still remain. We conduct our
own investigation, finding that process supervision significantly outper-
forms outcome supervision for training models to solve problems from the
challenging MATH dataset. Our process-supervised model solves 78% of
problems from a representative subset of the MATH test set. Additionally,
we show that active learning significantly improves the efficacy of process
supervision. To support related research, we also release PRM800K, the
complete dataset of 800,000 step-level human feedback labels used to train
our best reward model.

# 1
Introduction

Large language models are capable of solving tasks that require complex multi-
step reasoning by generating solutions in a step-by-step chain-of-thought format
(Nye et al., 2021; Wei et al., 2022; Kojima et al., 2022). However, even state-
of-the-art models are prone to producing...

# 2
Methods

We perform a comparison of outcome and process supervision, following a sim-
ilar methodology to Uesato et al. (2022). Outcome supervision can be provided
without humans, since all problems in the MATH dataset have automatically
checkable answers. In contrast, there is no simple ...

## 2.1
Scope

At each model scale, we use a single fixed model to generate all solutions. We
call this model the generator. We do not attempt to improve the generator with
reinforcement learning (RL). When we discuss outcome and process supervision,
we are specifically referring to the supervi...

## 2.2
Base Models

All large-scale models are finetuned from the base GPT-4 model (OpenAI, 2023).
This model has been pretrained solely to predict the next token; it has not been
pretrained with any Reinforcement Learning from Human Feedback (RLHF)
(Christiano et al., 2017). The small-scale base mo...

*[figure -- see caption above/below]*

*[figure -- see caption above/below]*

*[figure -- see caption above/below]*

*[figure -- see caption above/below]*

*[figure -- see caption above/below]*

*[figure -- see caption above/below]*

*[figure -- see caption above/below]*

*[figure -- see caption above/below]*

*[figure -- see caption above/below]*

*[figure -- see caption above/below]*

*[figure -- see caption above/below]*

*[figure -- see caption above/below]*

*Figure 1: A screenshot of the interface used to collect feedback for each step in
a solution.*

## 2.3
Generator

To make parsing individual steps easier, we train the generator to produce
solutions in a newline delimited step-by-step format. Specifically, we few-shot
generate solutions to MATH training problems, filter to those that reach the
correct final answer, and finetune the base mode...

## 2.4
Data Collection

To collect process supervision data, we present human data-labelers with step-
by-step solutions to MATH problems sampled by the large-scale generator.
Their task is to assign each step in the solution a label of positive, negative,
or neutral, as shown in Figure 1. A positive la...

## 2.5
Outcome-supervised Reward Models (ORMs)

We train ORMs following a similar methodology to Cobbe et al. (2021). We
uniformly sample a fixed number of solutions per problem from the generator,
and we train the ORM to predict whether each solution is correct or incorrect.
In practice, we usually determine correctness by au...

## 2.6
Process-supervised Reward Models (PRMs)

We train PRMs to predict the correctness of each step after the last token in
each step. This prediction takes the form of a single token, and we maximize the

*[figure -- see caption above/below]*

*Figure 2: Two solutions to the same problem, graded by the PRM. The solution
on the left is correct while the solution on the right is incorrect.
A green
background indicates a high PRM score, and a red background indicates a low
score. The PRM correctly identifies the mistake in the incorrect solution.*

log-likelihood of these target tokens during training. The PRM can therefore
be trained in a standard language model pipel...

*[table -- see caption above/below]*

*[figure -- see caption above/below]*

*Figure 3: A comparison of outcome-supervised and process-supervised reward
models, evaluated by their ability to search over many test solutions. Majority
voting is shown as a strong baseline. For N ≤1000, we visualize the variance
across many subsamples of the 1860 solutions we generated in total per problem.*

# 3
Large-scale Supervision

We train the large-scale PRM using the step-level labels in PRM800K. To ensure
the large-scale ORM baseline is as strong as possible, we train on 100 uniform
samples per problem from the generator. This means the ORM training set has
no overlap with PRM800K, and it is an order of...

*[figure -- see caption above/below]*

*[figure -- see caption above/below]*

*Figure 4: A comparison of different forms of outcome and process supervision.
Mean and standard deviation is shown across three seeds.*

# 4
Small-scale Synthetic Supervision

We find that the PRM outperforms the ORM at large-scale, but this result alone
paints an incomplete picture. To better compare outcome and process supervi-
sion, there are two confounding factors that must be isolated. First, the training
sets for the ORM and the PRM are not dire...

## 4.1
Process vs Outcome Supervision

We now conduct a direct comparison of outcome and process supervision. We
first sample between 1 and 200 solutions per problem from a small-scale genera-
tor. For each dataset, we provide three forms of supervision: process supervision
from PRMlarge, outcome supervision from PRMl...

## 4.2
Active Learning

Finally, we investigate the impact of active learning. We train a small-scale
reward model, PRMselector, on a single sample from each problem, and we use
this model to score 1000 samples per problem. To train each of our larger re-
ward models, we select N samples per problem suc...

*[table -- see caption above/below]*

*Table 1: We measure out-of-distribution generalization using recent STEM tests.
We evaluate the outcome-supervised RM, the process-supervised RM, and ma-
jority voting using 100 test samples per problem.*

# 5
OOD Generalization

To get some measure of out-of-distribution generalization, we evaluate our large-
scale ORM and PRM on a held-out set of 224 STEM questions, pulled from the
most recent AP Physics, AP Calculus, AP Chemistry, AMC10, and AMC12 ex-
ams. Since these tests were released after the pre-...

# 6
Discussion

## 6.1
Credit Assignment

One clear advantage of process supervision is that it provides more precise
feedback than outcome supervision.
A reward model trained with outcome
supervision faces a difficult credit-assignment task — to generalize well, it must
determine where an incorrect solution went wrong. ...

## 6.2
Alignment Impact

Process supervision has several advantages over outcome supervision related
to AI alignment. Process supervision is more likely to produce interpretable
reasoning, since it encourages models to follow a process endorsed by humans.
Process supervision is also inherently safer: it ...

## 6.3
Test Set Contamination

The test set of the MATH dataset contains problems that are discussed in
several online venues, and it is likely that some of these problems appear in
the pretraining dataset for our models. We attempted to remove all MATH
problems from our MathMix dataset using string-matching h...

# 7
Related Work

## 7.1
Outcome vs Process Supervision

In work closely related to our own, Uesato et al. (2022) compare the impact
of outcome and process supervision in the domain of grade school math. They
found that both methods led to similar final-answer error rates, and that process
supervision achieved those results with less d...

## 7.2
Synthetic Supervision

Similar to our work in Section 4, Gao et al. (2022) use a large reward model to
supervise the training of smaller models. They study the over-optimization that
occurs during RLHF, with experiments that require large quantities of human
preference data. To work around this challen...

## 7.3
Natural Language Reasoning

Several recent studies that have examined the reasoning ability of large language
models are implicitly relevant to our work. Lewkowycz et al. (2022) showed that
finetuning models on a large corpus of technical content led to significantly im-
proved performance on MATH. Wang et ...

# 8
Conclusion

We have shown that process supervision can be used to train much more reliable
reward models than outcome supervision in the domain of mathematical rea-
soning. We have also shown that active learning can be used to lower the cost of
human data collection by surfacing only the mo...

# Acknowledgements

We thank Joshua Achiam, Mark Chen, Jonathan Gordon, Dan Hendrycks,
Lukasz Kaiser, Oleg Murk, Ben Sokolowsky, Francis Song, and Jonathan Uesato
for valuable feedback and thoughtful discussions; Giambattista Parascandolo
and Daniel Selsam for their contributions to the MathMix data...

# References

OpenAI. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

# A
MathMix

Similar to Lewkowycz et al. (2022) we construct a large-scale dataset of high-
quality math-relevant tokens for use in a lightweight pretraining stage, before
finetuning on comparably smaller datasets like MATH and PRM800K. This
dataset, which we call MathMix, has two main differ...

*[table -- see caption above/below]*

*Table 2: MathMix dataset components.*

# B
PRM800K

We collected 1,085,590 step-level labels over 101,599 solution samples.
We
present the whole unfiltered dataset as PRM800K. During training we discard
labels used for quality control, as well as any step-level labels for which the
labeler was unable to complete the task. The filt...

*[table -- see caption above/below]*

*Table 3: Distribution of positive/negative steps/solutions.*

# C
Evaluation

As we scaled up the project, we began having to collect labels on multiple
solutions for the same training problem. In order to avoid the risk of over-fitting
on the 7,500 MATH training problems, we expanded the training set to include
4,500 MATH test split problems. We therefore...

*[figure -- see caption above/below]*

*[figure -- see caption above/below]*

*Figure 5: Two histograms comparing the distribution of problem difficulty levels
and subjects in both the original MATH test set and in our 500 problem test
subset.*

# D
Labelling Instructions

Labelers were tasked to look at steps in a solution and label each one as posi-
tive, negative, or neutral. A step is considered neutral if it is appropriate in
context, reasonable, correct, and contains only computations that can be veri-
fied easily. A step is positive if it is...

# E
ORM Training Details

We train outcome-supervised reward models in the same manner as token-level
verifiers from Cobbe et al. (2021), with a few subtle differences to hyperparam-
eters. In particular, we only train for a single epoch on each dataset of model
samples and reward model labels, without dr...

# F
PRM Details

## F.1
Training

We train our PRMs by fine-tuning the MathMix model to predict the probability
of positive, negative, and neutral labels given a solution prefix ending in one of
our labeled steps. We sweep over hyperparameters using a dataset containing
the first ∼10% of PRM800K. Fine-tuning an L...

## F.2
Scoring

There are multiple ways of using the PRM to score solutions. In general, we
produce a single solution-level score by performing a reduction over step-level
scores, where the step-level score is the probability that the step’s label is pos-
itive. This involves two specific implem...

*[table -- see caption above/below]*

*Table 4: Best-of-1860 test performance using the PRM with four different scor-
ing strategies.*

# G
Difficulty Breakdown

We show performance of our ORM and PRM on each quintile of the MATH
dataset. We determine quintiles based on the pass rate under the generator.
It is interesting to note that the performance gap is not only apparent on high
difficulty problems: it is in fact apparent across all d...

*[figure -- see caption above/below]*

*Figure 6: A breakdown of ORM vs PRM performance by problem difficulty.*

# H
Synthetic Supervision Details

We can use PRMlarge to provide either outcome or process supervision for
smaller models.
We determine the labels for individual steps based on the
step-level probabilities outputted by PRMlarge. To do this, we set an arbitrary
threshold: any step that PRMlarge assigns a negative ...

# I
PRM Visualizations

All examples shown come from the large-scale generator (GPT-4).
We note
the pass-rate under the generator to give some sense of the difficulty of these
problems.

## I.1
True Positives

These cherry-picked examples show the best-of-1860 solution from the generator
as ranked by the large-scale PRM.

Problem 1. Generator pass-rate: 0.1%. This challenging trigonometry problem
requires applying several identities in a not-at-all obvious succession.
Most
solution attem...

*Problem 2. Generator pass-rate: 5.8%. In step 7 and 8, the generator starts
performing guess-and-check. This is a common place the model might hallu-
cinate, by claiming a particular guess is successful when it isn’t. In this case,
the reward model verifies each step and determines that the chain-of-thought is
correct.*

*Problem 3. Generator pass-rate: 1.7%. The generator successfully applies sev-
eral trigonometric identities to simplify the expression.*

*[figure -- see caption above/below]*

## I.2
True Negatives

Problem 5.
Generator pass-rate: 4.5%.
The generator attempts to use the
difference of squares formula in step 12 on an expression that isn’t in fact a
difference of squares. The reward model catches this mistake.

*[figure -- see caption above/below]*

Problem 6. Generator pass-rate: 93.5%. In step 7, the generator make...

## I.3
False Positives

Problem 9. Generator pass-rate: 18.5%. The generator makes a subtle counting
error in step 9. On the surface, it appears reasonable to claim that there are 5
ways to exchange the same colored ball since there are 5 colors. However, this
undercounts by a factor of 2, since Bob has...

---

<!-- ===== 05_Madaan_SelfRefine_NeurIPS2023.pdf ===== -->

# SELF-REFINE:
Iterative Refinement with Self-Feedback

# Abstract

Like humans, large language models (LLMs) do not always generate the best
output on their first try. Motivated by how humans refine their written text, we
introduce SELF-REFINE, an approach for improving initial outputs from LLMs
through iterative feedback and refinement. The main idea is to generate an initial
output using an LLM; then, the same LLM provides feedback for its output and
uses it to refine itself, iteratively. SELF-REFINE does not require any supervised
training data, additional training, or reinforcement learning, and instead uses a
single LLM as the generator, refiner and the feedback provider. We evaluate
SELF-REFINE across 7 diverse tasks, ranging from dialog response generation
to mathematical reasoning, using state-of-the-art (GPT-3.5 and GPT-4) LLMs.
Across all evaluated tasks, outputs generated with SELF-REFINE are preferred by
humans and automatic metrics over those generated with the same LLM using
conventional one-step generation, improving by ∼20% absolute on average in task
performance. Our work demonstrates that even state-of-the-art LLMs like GPT-4
can be further improved at test-time using our simple, standalone approach.1.

# 1
Introduction

Although large language models (LLMs) can generate coherent outputs, they often fall short in
addressing intricate requirements. This mostly includes tasks with multifaceted objectives, such
as dialogue response generation, or tasks with hard-to-define goals, such as enhancing pr...

*[figure -- see caption above/below]*

*Figure 1: Given an input ( 0⃝), SELF-REFINE starts by generating an output and passing it back to the
same model M to get feedback ( 1⃝). The feedback is passed back to M, which refines the previously
generated output ( 2⃝). Steps ( 1⃝) and ( 2⃝) iterate until a stopping condition is met. SELF-REFINE is
instantiated with a language model such as GPT-3.5 and does not involve human assistance.*

# 2
Iterative Refinement with SELF-REFINE

Given an input sequence, SELF-REFINE generates an initial output, provides feedback on the output,
and refines the output according to the feedback. SELF-REFINE iterates between feedback and
refinement until a desired condition is met. SELF-REFINE relies on a suitable language mo...

# (b) FEEDBACK fb

# (a) Dialogue: x, yt

# (c) REFINE yt+1

Response (refined): That's
great to hear (...) ! It's
a fun sport requiring
quick reflexes and good
hand-eye coordination.
Have you played before, or
are you looking to learn?

User: I am interested
in playing Table
tennis.

Engaging: Provides no
information about table
tennis or how...

# (d) Code optimization: x, yt

# (e) FEEDBACK fb

# (f) REFINE yt+1

Code (refined)
def sum_faster(n):
return (n*(n+1))//2

Generate sum of 1, ..., N
def sum(n):
res = 0
for i in range(n+1):
res += i
return res

This code is slow as
it uses brute force.
A better approach is
to use the formula
... (n(n+1))/2.

Figure 2: Examples of SELF-REFINE: an initia...

# Algorithm 1 SELF-REFINE algorithm

Require: input x, model M, prompts {pgen, pfb, prefine}, stop condition stop(·)
1: y0 = M(pgen∥x)
▷Initial generation (Eqn. 1)
2: for iteration t ∈0, 1, . . . do
3:
fbt = M (pfb∥x∥yt)
▷Feedback (Eqn. 2)
4:
if stop(fbt, t) then
▷Stop condition
5:
break
6:
else
7:
yt+1 = M (prefine...

*Figure 3: The SELF-REFINE algorithm. See (§2) for a discussion of each component.*

# 3
Evaluation

We evaluate SELF-REFINE on 7 diverse tasks: Dialogue Response Generation (Appendix M; Mehri
and Eskenazi, 2020), Code Optimization (Appendix N; Madaan et al., 2023), Code Readability
Improvement (Appendix L; Puri et al., 2021), Math Reasoning (Appendix O; Cobbe et al., 2021),
Sen...

## 3.1
Instantiating SELF-REFINE

We instantiate SELF-REFINE following the high-level description in Section 2. The FEEDBACK-
REFINE iterations continue until the desired output quality or task-specific criterion is reached, up to a
maximum of 4 iterations. To make our evaluation consistent across different model...

*[table -- see caption above/below]*

*Table 1: SELF-REFINE results on various tasks using GPT-3.5, ChatGPT, and GPT-4 as base LLM.
SELF-REFINE consistently improves LLM. Metrics used for these tasks are defined in Section 3.2.*

## 3.2
Metrics

We report three types of metrics:

## 3.3
Results

Table 1 shows our main results:

SELF-REFINE consistently improves over base models across all model sizes, and additionally
outperforms the previous state-of-the-art across all tasks. For example, GPT-4+SELF-REFINE
improves over the base GPT-4 by 8.7% (absolute) in Code Optimizati...

# 4
Analysis

The three main steps of SELF-REFINE are FEEDBACK, REFINE, and repeating them iteratively. In this
section, we perform additional experiments to analyze the importance of each of these steps.

*[table -- see caption above/below]*

*Table 2: Prompting to generate generic feedback (or having the model generate no feedback at
all) leads to reduced scores, indicating the importance of the FEEDBACK step of SELF-REFINE.
These experiments were performed with ChatGPT (Code Optimization and Sentiment Reversal) and
GPT-3.5 (Acronym Generation), and metrics used are defined in Section 3.2.*

The impact of the feedback quality
Feedback quality plays a crucial role in SELF-REFINE. T...

*[figure -- see caption above/below]*

*[table -- see caption above/below]*

*Figure 4: Left: Iteration-wise score improvements. Early iterations significantly improve output
quality, and scores generally keep improving with more iterations. Right: SELF-REFINE Performance
improvements with iterations. Most gains(∆) are in the initial iterations for both Code Opt. and Senti-
ment Reversal. The numbers are averaged over ChatGPT, GPT-3.5, and GPT-4. Task abbreviations:
C. Opt. (Code Optimiz.), S. Rev. (Sentiment Reversal), C. Gen. (Constrained Generation).*

*Figure 5: Comparison of code generated by Madaan et al. (2023) (left) and the output after applying
SELF-REFINE (right). The initial code by the baseline, which is nearly identical to the slower input
program, fails to improve the efficiency and merely alters the logic for reading input. SELF-REFINE
first generates feedback that diagnoses that This code is slow because it is using six nested loops to
iterate through all possible combinations of coins to pay the amount, and suggests that a more efficient
approach would be .... SELF-REFINE then uses this feedback to generate the revised code (right),
reducing the time complexity to O(amount ∗coins). The full example is provided in Appendix H*

# 5
Related work

Leveraging human- and machine-generated natural language (NL) feedback for refining outputs has
been effective for a variety of tasks, including summarization (Scheurer et al., 2022), script generation
(Tandon et al., 2021), program synthesis (Le et al., 2022a; Yasunaga and Liang...

*[table -- see caption above/below]*

*Table 3: A comparison of SELF-REFINE to closely related prior refinement approaches.*

# 6
Limitations and Discussion

The main limitation of our approach is that the base models need to have sufficient few-shot modeling
or instruction-following abilities, in order to learn to provide feedback and to refine in an in-context
fashion, without having to train supervised models and rely on supervised...

# 7
Conclusion

We present SELF-REFINE: a novel approach that allows large language models to iteratively provide
self-feedback and refine their own outputs. SELF-REFINE operates within a single LLM, requiring
neither additional training data nor reinforcement learning. We demonstrate the simpli...

# References

OpenAI. 2022. Model index for researchers. Blogpost.

OpenAI. 2023. Gpt-4 technical report.

# A
Evaluation Tasks

Table 4 lists the tasks in our evaluation, and examples from each task.

# Task and Description

# Sample one iteration of FEEDBACK-REFINE

# Sentiment Reversal

x: The food was fantastic...”

yt: The food was disappointing...”

Rewrite reviews to reverse sentiment.

Dataset:

fb: Increase negative sentiment

Dataset: (Zhang et al., 2015) 1000 review pas-
sages

yt+1: The food was utterly terrible...”

x: What’s the best way to cook pasta?”

# Dialogue Response Generation

yt: The best way to cook pasta is to...”

Produce rich conversational responses.

(Mehri and Eskenazi, 2020) 372 conv.

Dataset:

fb: Make response relevant, engaging, safe

Boil water, add salt, and cook pasta...”

yt+1:

# Code Optimization

x: Nested loop for matrix product

Enhance Python code efficiency

yt: NumPy dot product function

(Madaan et al., 2023): 1000 programs

Dataset:

fb: Improve time complexity

yt+1: Use NumPy’s optimized matmul function

# Code Readability Improvement

x: Unclear variable names, no comments

Refactor Python code for readability.

yt: Descriptive names, comments

(Puri et al., 2021) 300 programs∗

Dataset:

fb: Enhance variable naming; add comments

yt+1: Clear variables, meaningful comments

# Math Reasoning

x: Olivia has $23, buys 5 bagels at $3 each”

Solve math reasoning problems.

yt: Solution in Python

Dataset:

(Cobbe et al., 2021) 1319 questions

fb: Show step-by-step solution

Solution with detailed explanation

yt+1:

# Acronym Generation

Radio Detecting and Ranging”

x :

RDR

Generate acronyms for a given title

yt:

t
fb :

Dataset:

be context relevant; easy pronunciation

(Appendix Q) 250 acronyms

RADAR”

yt+1:

# Constrained Generation

beach, vacation, relaxation

x:

Generate sentences with given keywords.

yt: During our beach vacation...

Dataset:

(Lin et al., 2020) 200 samples

fb: Include keywords; maintain coherence

yt+1: .. beach vacation was filled with relaxation

Table 4: An overview of the tasks which we evaluate ...

# B
Broader Related Work

Compared to a concurrent work, Reflexion (Shinn et al., 2023), our approach involves correction
using feedback, whereas their setup involves finding the next best solution in planning using ReAct.
While ReAct and Reflexion provide a free-form reflection on whether a step was exec...

*See Table 5 for an additional detailed comparison of related work.*

*[table -- see caption above/below]*

*Table 5: Summary of related approaches. Reinforcement learning approaches are shown in purple*

# C
Human Evaluation

The A/B evaluation in our study was conducted by the authors, where a human judge was presented
with an input, task instruction, and two candidate outputs generated by the baseline method and
SELF-REFINE. The setup was blind, i.e., the judges did not know which outputs were gener...

*[table -- see caption above/below]*

*Table 6: Relative improvement of SELF-REFINE in A/B evaluations across different tasks. The values
represent normalized preferences, which correspond to the proportion of times the output generated
by SELF-REFINE was selected as better aligned with the task instruction over the baseline method.
The evaluation was conducted for 150 examples for each dataset. The judges were not aware of the
method that generated each sample.*

# D
GPT-4 Evaluation

In light of the impressive achievements of GPT-4 in assessing and providing reasoning for complex
tasks, we leverage its abilities for evaluation in SELF-REFINE. The approach involves presenting
tasks to GPT-4 in a structured manner, promoting the model’s deliberation on the task...

# Listing 1 Prompt for GPT-4 evaluation of Sentiment Reversal.

f"""Which review is aligned with the sentiment {target_sentiment}?
Review A: {review_a}
Review B: {review_b}.
Pick your answer from ['Review A', 'Review B', 'both', 'neither']. Generate a
short explanation for your choice first. Then, generate 'The more aligned
review is A' or 'T...

# Listing 2 Prompt for GPT-4 evaluation of Acronym Generation.

f"""Title: {title}
Acronym A: {acronym_a}
Acronym B: {acronym_b}
Pick the better acronym for the given title. The acronyms should be compared based
on the following criteria:
�→
* Ease of pronunciation.
* Ease of spelling.
* Relation to title.
* Positive connotation.
Generate you...

# Listing 3 Prompt for GPT-4 evaluation of Dialogue Response Generation.

f"""Which response is better given this context: {context}?
Response A: {response_a}
Response B: {response_b}.
Pick your answer from ['Response A', 'Response B', 'both', 'neither']. Generate a
short explanation for your choice first. Then, generate 'The better response
is A' or '...

# E
Model Key

We use terminology here: https://platform.openai.com/docs/models/gpt-3-5

# F
Comparison of SELF-REFINE with State-of-the-art of Few-Shot Learning
Models and Fine-Tuned Baselines

In this section, we present a comprehensive comparison of the performance of SELF-REFINE with
other few-shot models and fine-tuned baselines across a range of tasks, including mathematical
reasoning and programming tasks. Tables 8 and 7 display the performance of these models on ...

*[table -- see caption above/below]*

*Table 7: Performance comparison of models on math reasoning (Math Reasoning).*

*Table 8: Performance comparison of various models on the PIE dataset in terms of the percentage
of programs optimized (%OPT). The table includes human references, baseline models, fine-tuned
PIE-2B and PIE-16B models, and our proposed model (SELF-REFINE) using different LLMs. Notably,
SELF-REFINE achieves superior performance while using only 4 samples at most, significantly fewer
than the 16 and 32 samples employed by other models. Scalene, an off-the-shelf optimizer, uses
instruction tuning with Codex and serves as a comparison point.*

# G
Evaluation of Vicuna-13b

We also experiment with Vicuna-13b (Chiang et al., 2023), a version of LLaMA-13b (Touvron et al.,
2023) fine-tuned on conversations sourced from the web. Vicuna-13b was able to consistently follow
the task initialization prompt. However, it struggled to follow the prompts intende...

*[figure -- see caption above/below]*

*Figure 6: Preference for the outputs generated by our method (SELF-REFINE), the multiple-sample
baseline (MULTI), and ties (ties).*

*[table -- see caption above/below]*

*Table 9: SELF-REFINE results on Math Reasoning using GPT-3.5, ChatGPT, and GPT-4 as base
LLM with Oracle feedback.*

# H
Additional Analysis

## H.1
Using Oracle Feedback

We experimented with Oracle Feedback following Welleck et al. (2022). This method uses correctness
information to guide model refinement, only progressing to REFINE stage if the current answer is
incorrect. This adjustment notably enhanced performance in the Math Reasoning task, ...

*[table -- see caption above/below]*

*Table 10: Acronym generation results across iterations, showcasing how improvements in certain as-
pects (e.g., pronunciation and spelling) can be accompanied by losses in others, leading to fluctuating
overall performance in multi-aspect feedback tasks like Acronym Generation.*

*Table 11: Error analysis for Dialogue Response Generation: When the feedback is not useful, a large
majority is not specific or incorrect.*

*Table 12: On the Dialogue Response Generation task, SELF-REFINE can ignore good feedback but in
a majority of cases, it is robust to bad feedback and ignores bad feedback.*

# I
Beyond Benchmarks

SELF-REFINE demonstrates its iterative feedback and refinement capabilities in the context of
website layout generation. ChatGPT initially produces a rudimentary layout for a given topic, and
then uses the FEEDBACK to suggest specific, actionable improvements, as demonstrated in ...

*[figure -- see caption above/below]*

*Figure 7: Initial web layout generated by our model for a fictional ice cream parlor.*

*Figure 8: Refined web layout after applying model feedback. The feedback included changing the
background color to light blue (#6f2ff), increasing the heading font size to 48px, adding an icon
before the welcome text, enhancing the content with an additional paragraph, increasing the button
text size to 24px, and updating the button color to #9933.*

*Figure 9: Initial web layout generated by our model for a page on photosynthesis.*

*Figure 10: Refined web layout after applying model feedback. The feedback included increasing
the text font size to 18px for better readability, adding more information about the benefits of
photosynthesis, removing the unnecessary margin-top from the header, and adding a ruler or divider
below the header to separate it from the image.*

# J
Statistical Confidence Intervals

*[table -- see caption above/below]*

*Table 13: SELF-REFINE results from table 1 with Wilson confidence interval (at 95% confidence
interval) and statistical significance. On various tasks using GPT-3.5, ChatGPT, and GPT-4 as
base LLM, SELF-REFINE consistently improves LLM. Metrics used for these tasks are defined in
Section 3.2 as follows: Math Reasoning uses the solve rate; Code Optimization uses the percentage
of programs optimized; and Sentiment Reversal, Dialogue Response and Acronym Gen use a GPT-
4-based preference evaluation, which measures the percentage of times outputs from the base or
enhanced models were selected, with the rest categorized as a tie. Constrained Gen uses the coverage
percentage. Gains over Base, that are statistically significant based on these confidence intervals are
marked **

*Table 13 shows results from Table 1 with Wilson confidence interval (Brown et al., 2001) (at α=
99% confidence interval) and statistical significance. Gains that are statistical significance based on
these confidence intervals are marked with an asterisk. We find that nearly all of GPT-4 gains are
statistically significant, ChatGPT gains are significant for 4 out of 7 datasets, and GPT-3.5 gains are
significant for 3 out of 7 datasets.*

# K
New Tasks

Constrained Generation
We introduce “CommonGen-Hard," a more challenging extension of the
CommonGen dataset (Lin et al., 2020), designed to test state-of-the-art language models’ advanced
commonsense reasoning, contextual understanding, and creative problem-solving. CommonGen-
Ha...

# L
Code Readability

Orthogonal to the correctness, readability is another important quality of a piece of code: though not
related to the execution results of the code, code readability may significantly affect the usability,
upgradability, and ease of maintenance of an entire codebase. In this sect...

## L.1
Method

Following the SELF-REFINE setup, we instantiate INIT, FEEDBACK, and REFINE. The INIT is a no-op
— we directly start by critiquing the code with FEEDBACK and applying the changes with REFINE.

Starting from an initial piece of code y0, we first critique, c1 = critique(y0), and then ...

## L.2
Experiments

Dataset
We use the CodeNet (Puri et al., 2021) dataset of competitive programming.5 For our
purpose, these are hard-to-read multi-line code snippets. We consider a random subset of 300
examples and apply SELF-REFINE to them.

We also ask human annotators to edit a 60-example subset...

*[table -- see caption above/below]*

*Table 14: Human v.s. SELF-REFINE performance on 60-example subset. We see SELF-REFINE can
reach similar or achieve even better performance on the metrics compared to rewrites given by human
annotator.*

*[figure -- see caption above/below]*

*[figure -- see caption above/below]*

*[figure -- see caption above/below]*

*(a) Meaningful variable ratio across
different SELF-REFINE iterations.
(b) Comment per line ratio across
different SELF-REFINE iterations.
(c) Number of function units across
different SELF-REFINE iterations.*

*Figure 11: Evaluation on code readability task with SELF-REFINE across multiple metrics*

# Example

# M
Dialogue Response Generation

Open-domain dialogue response generation is a complex task that requires a system to generate
human-like responses to a wide range of topics. Due to the open-ended nature of the task, it is

## Starting Code: v0

print((int((int(eval(input()))+1)/2)))

# Code v1

print(
(int(
(int(eval(input())) + 1)
/ 2
)
)

# Code v2

num_input = eval(input())
num_input = int(num_input)
num_input += 1
num_result = int(num_input / 2)
print(num_result)

Figure 12: SELF-REFINE iterations over a piece of densely composed code. Through out the
iterations, the model first try to indent out the code and ends up rewriti...

## M.1
Modules

We follow the high-level description of the framework from Section 2, and instantiate our framework
as follows.

Init
This is the first step in performing the task. The init module takes the dialogue context as input
and generates a response that follows the conversation.

Feedback
W...

*Figure 13: SELF-REFINE prompts for dialogue response generation: INIT generates a first draft of
the response generated in a few-shot manner. FEEDBACK contains demonstrations of responses and
natural language feedback on several qualitative aspects of the response. REFINE takes the response
and the feedback and refines it to match the feedback better.*

## M.2
Setup and Experiments

Model and Baseline
We establish a natural baseline for our approach by using the model directly,
without any feedback, which we refer to as INIT. Our implementation of SELF-REFINE employs a
few-shot setup, where each module (INIT, FEEDBACK, ITERATE) is implemented as few-shot pro...

*[table -- see caption above/below]*

*Table 15: Human evaluation results for dialogue response generation*

# N
Code Optimization

Performance-Improving Code Edits or PIE (Madaan et al., 2023) focuses on enhancing the efficiency
of functionally correct programs. The primary objective of PIE is to optimize a given program by
implementing algorithmic modifications that lead to improved runtime performance.

Give...

*Table 16: Main Results and Ablation Analysis*

# O
Math Reasoning

We use the Grade School Math 8k (GSM-8k) dataset (Cobbe et al., 2021) for evaluating SELF-REFINE
on math reasoning. In the context of grade school mathematics, SELF-REFINE aims to enable LLMs
to iteratively refine their mathematical problem-solving outputs based on introspective ...

*[figure -- see caption above/below]*

*Figure 14: Improvements in accuracy on the GSM-8k math reasoning benchmark as a function of the
# of iterations of SELF-REFINE.*

# P
Sentiment Reversal

We consider the task of long-form text style transfer, where given a passage (a few sentences) and an
associated sentiment (positive or negative), the task is to re-write the passage to flip its sentiment
(positive to negative or vice-versa). While a large body of work on style t...

## P.1
Details

Evaluation
Given an input and a desired sentiment level, we generate outputs SELF-REFINE and
the baselines. Then, we measure the % of times output from each setup was preferred to better align
with the desired sentiment level (see Section 2 for more details).

We also experiment wi...

# Q
Acronym Generation

Good acronyms provide a concise and memorable way to communicate complex ideas, making them
easier to understand and remember, ultimately leading to more efficient and effective communication.
Like in email writing, acronym generation also requires an iterative refinement process...

*[table -- see caption above/below]*

*Table 18: Comparison of acronyms for input = “Sequence to Sequence Learning with Neural
Networks”*

# R
Constrained Generation

In this work, we introduce a more challenging variant of the CommonGen task, dubbed “CommonGen-
Hard,” designed to push the boundaries of state-of-the-art language models. CommonGen-Hard
requires models to generate coherent and grammatically correct sentences incorporating 20-30
...

# S
Prompts

We include all the prompts used in the experiments in Figures 16-35:

Recall that the Base LLM requires a generation prompt pgen with input-output pairs ⟨xi, yi⟩, the
feedback module requires a feedback prompt pfb with input-output-feedback triples ⟨xi, yi, fbi⟩, and
the refinement...

*[figure -- see caption above/below]*

*Figure 15: A comparison of SELF-REFINE and direct generation with GPT-3.5 on CommonGen-
Hard.*

*Figure 16: Initial generation prompt for Acronym Generation*

*Figure 17: FEEDBACK prompt for Acronym Generation*

*Figure 18: REFINE prompt for Acronym Generation*

*Figure 19: Initial generation prompt for Code Optimization*

*Figure 20: FEEDBACK prompt for Code Optimization*

*Figure 21: REFINE prompt for Code Optimization*

*Figure 22: FEEDBACK prompt for Code Readability*

*Figure 23: REFINE prompt for Code Readability*

*Figure 24: Initial generation prompt for Constrained Generation (truncated)*

*Figure 25: FEEDBACK prompt for Constrained Generation (truncated).*

*Figure 26: REFINE prompt for Constrained Generation (truncated)*

*Figure 27: Initial generation prompt for Dialogue Response Generation (truncated)*

*Figure 28: FEEDBACK prompt for Dialogue Response Generation (truncated)*

*Figure 29: REFINE prompt for Dialogue Response Generation (truncated)*

*Figure 30: Initial generation prompt for Math Reasoning*

*Figure 31: FEEDBACK prompt for Math Reasoning*

*Figure 32: REFINE prompt for Math Reasoning*

*Figure 33: Initial generation prompt for Sentiment Reversal (truncated)*

*Figure 34: FEEDBACK prompt for Sentiment Reversal (truncated).*

*Figure 35: REFINE prompt for Sentiment Reversal (truncated).*

---

<!-- ===== 06_Schick_Toolformer_NeurIPS2023.pdf ===== -->

# Toolformer: Language Models Can Teach Themselves to Use Tools

# Timo Schick
Jane Dwivedi-Yu
Roberto Dessì†
Roberta Raileanu
Maria Lomeli
Luke Zettlemoyer
Nicola Cancedda
Thomas Scialom

# Abstract

The New England Journal of Medicine is a registered 
trademark of [QA(“Who is the publisher of The New 
England Journal of Medicine?”) → Massachusetts 
Medical Society] the MMS.

Language models (LMs) exhibit remarkable
abilities to solve new tasks from just a few
examples or textual instructions, especially at
scale. They also, paradoxically, struggle with
basic functionality, such as arithmetic or fac-
tual lookup, where much simpler and smaller
models excel.
In this paper, we show that
LMs can teach themselves to use external tools
via simple APIs and achieve the best of both
worlds.
We introduce Toolformer, a model
trained to decide which APIs to call, when to
call them, what arguments to pass, and how to
best incorporate the results into future token
prediction. This is done in a self-supervised
way, requiring nothing more than a handful of
demonstrations for each API. We incorporate
a range of tools, including a calculator, a Q&A
system, a search engine, a translation system,
and a calendar. Toolformer achieves substan-
tially improved zero-shot performance across
a variety of downstream tasks, often competi-
tive with much larger models, without sacrific-
ing its core language modeling abilities.

Out of 1400 participants, 400 (or [Calculator(400 / 1400)
→ 0.29] 29%) passed the test.

The name derives from “la tortuga”, the Spanish word for 
[MT(“tortuga”) → turtle] turtle.

The Brown Act is California’s law [WikiSearch(“Brown 
Act”) → The Ralph M. Brown Act is an act of the 
California State Legislature that guarantees the public's
right to attend and participate in meetings of local 
legislative bodies.] that requires legislative bodies, like 
city councils, to hold their meetings open to the public.

Figure 1: Exemplary predictions of Toolformer. The
model autonomously decides to call different APIs
(from top to bottom: a question answering system,
a calculator, a machine translation system, and a
Wikipedia search engine) to obtain information that is
useful for completing a piece of text.

# 1
Introduction

A simple way to overcome these limitations of
today’s language models is to give them the abil-
ity to use external tools such as search engines,
calculators, or calendars. However, existing ap-
proaches either rely on large amounts of human
annotations (Komeili et al., 2022; Tho...

*[figure -- see caption above/below]*

*Figure 2: Key steps in our approach, illustrated for a question answering tool: Given an input text x, we first
sample a position i and corresponding API call candidates c1
i , c2
i , . . . , ck
i . We then execute these API calls and
filter out all calls which do not reduce the loss Li over the next tokens. All remaining API calls are interleaved
with the original text, resulting in a new text x∗.*

# 2
Approach

Our aim is to equip a language model M with the
ability to use different tools by means of API calls.
We require that inputs and outputs for each API
can be represented as text sequences. This allows
seamless insertion of API calls into any given text,
using special tokens to mar...

*Figure 3: An exemplary prompt P(x) used to generate
API calls for the question answering tool.*

# 3
Tools

# 4
Experiments

We explore a variety of tools to address different
shortcomings of regular LMs. The only constraints
we impose on these tools is that (i) both their inputs
and outputs can be represented as text sequences,
and (ii) we can obtain a few demonstrations of
their intended use. Concret...

## 4.1
Experimental Setup

Dataset Generation
Throughout all of our ex-
periments, we use a subset of CCNet (Wenzek et al.,
2020) as our language modeling dataset C and GPT-
J (Wang and Komatsuzaki, 2021) as our language
model M. To reduce the computational cost of
annotating C with API calls, we define he...

*[table -- see caption above/below]*

*Table 1: Examples of inputs and outputs for all APIs used.*

*Table 2: Number of examples with API calls in C∗for
different values of our filtering threshold τf.*

## 4.2
Downstream Tasks

to make sure that API calls happen close to where
the information provided by the API is actually
helpful for the model. The thresholds τs and τf are
chosen individually for each tool to ensure a suffi-
ciently larger number of examples; see Appendix A
for details. Table 2 shows ...

*[table -- see caption above/below]*

*Table 3: Results on subsets of LAMA. Toolformer uses
the question answering tool for most examples, clearly
outperforming all baselines of the same size and achiev-
ing results competitive with GPT-3 (175B).*

### 4.2.1
LAMA

*[table -- see caption above/below]*

We evaluate our models on the SQuAD, Google-
RE and T-REx subsets of the LAMA benchmark
(Petroni et al., 2019). For each of these subsets, the
task is to complete a short statement with a miss-
ing fact (e.g., a date or a place). As LAMA was
originally designed to evaluate masked...

*Table 4:
Results for various benchmarks requiring
mathematical reasoning. Toolformer makes use of the
calculator tool for most examples, clearly outperform-
ing even OPT (66B) and GPT-3 (175B).*

### 4.2.3
Question Answering

We look at Web Questions (Berant et al., 2013),
Natural Questions (Kwiatkowski et al., 2019) and
TriviaQA (Joshi et al., 2017), the three question an-
swering datasets considered by Brown et al. (2020).
For evaluation, we check whether the first 20 words
predicted by a model cont...

### 4.2.2
Math Datasets

We test mathematical reasoning abilities on ASDiv
(Miao et al., 2020), SVAMP (Patel et al., 2021) and
the MAWPS benchmark (Koncel-Kedziorski et al.,
2016). We again account for the fact that we test
all models in a zero-shot setup by using a more
lenient evaluation criterion: As ...

*[table -- see caption above/below]*

*[table -- see caption above/below]*

*Table 5: Results for various question answering dataset.
Using the Wikipedia search tool for most examples,
Toolformer clearly outperforms baselines of the same
size, but falls short of GPT-3 (175B).*

*Table 6: Results on MLQA for Spanish (Es), German
(De), Hindi (Hi), Vietnamese (Vi), Chinese (Zh) and
Arabic (Ar). While using the machine translation tool
to translate questions is helpful across all languages,
further pretraining on CCNet deteriorates performance;
consequently, Toolformer does not consistently outper-
form GPT-J. The final two rows correspond to models
that are given contexts and questions in English.*

### 4.2.4
Multilingual Question Answering

We evaluate Toolformer and all baseline models
on MLQA (Lewis et al., 2019), a multilingual
question-answering benchmark. A context para-
graph for each question is provided in English,
while the question can be in Arabic, German, Span-
ish, Hindi, Vietnamese, or Simplified Chine...

### 4.2.5
Temporal Datasets

To investigate the calendar API’s utility, we eval-
uate all models on TEMPLAMA (Dhingra et al.,
2022) and a new dataset that we call DATESET.
TEMPLAMA is a dataset built from Wikidata that
contains cloze queries about facts that change with
time (e.g., “Cristiano Ronaldo plays f...

*[table -- see caption above/below]*

*[table -- see caption above/below]*

*Table 8: Perplexities of different models on WikiText
and our validation subset of CCNet. Adding API calls
comes without a cost in terms of perplexity for lan-
guage modeling without any API calls.*

*Table 7: Results for the temporal datasets. Toolformer
outperforms all baselines, but does not make use of the
calendar tool for TEMPLAMA.*

## 4.4
Scaling Laws

We investigate how the ability to ask external tools
for help affects performance as we vary the size
of our LM. To this end, we apply our approach
not just to GPT-J, but also to four smaller mod-
els from the GPT-2 family (Radford et al., 2019),
with 124M, 355M, 775M and 1.6B pa...

## 4.3
Language Modeling

In addition to verifying improved performance on
various downstream tasks, we also want to ensure
that language modeling performance of Toolformer
does not degrade through our finetuning with API
calls.
To this end, we evaluate our models on
two language modeling datasets: WikiTe...

# 5
Analysis

Decoding Strategy
We investigate the effect of
our modified decoding strategy introduced in Sec-
tion 4.2, where instead of always generating the

*[figure -- see caption above/below]*

*Figure 4: Average performance on LAMA, our math benchmarks and our QA benchmarks for GPT-2 models of
different sizes and GPT-J finetuned with our approach, both with and without API calls. While API calls are not
helpful to the smallest models, larger models learn how to make good use of them. Even for bigger models, the
gap between model predictions with and without API calls remains high.*

*[table -- see caption above/below]*

most likely token, we generate the <API> token
if it is one of the k most likely tokens. Table 9
shows performance on the T-REx subset ...

*Table 9: Toolformer results on the T-REx subset of
LAMA and on WebQS for different values of k used
during decoding. Numbers shown are overall perfor-
mance (All), performance on the subset where the
model decides to make an API call (AC) and all re-
maining examples (NC), as well as the percentage of
examples for which the model decides to call an API
(%).*

# 6
Related Work

Language Model Pretraining
There are various
approaches that augment language models with
some form of additional textual information during
pretraining, including various forms of metadata
(Keskar et al., 2019), HTML tags (Aghajanyan
et al., 2021), Wikipedia markup (Schick et al...

*[table -- see caption above/below]*

*Table 10: Examples of API calls for different tools, sorted by the value of L−
i −L+
i that is used as a filtering
criterion. High values typically correspond to API calls that are intuitively useful for predicting future tokens.*

# 7
Limitations

# References

While our approach enables LMs to learn how to
use a variety of tools in a self-supervised way, there
are some clear limitations to what can be achieved
with our method in its current form. One such limi-
tation is the inability of Toolformer to use tools in a
chain (i.e., using ...

# 8
Conclusion

We have introduced Toolformer, a language model
that learns in a self-supervised way how to use
different tools such as search engines, calculators,
and translation systems via simple API calls. This
is done by finetuning on a large number of sampled
API calls that are filtered b...

# A
API Details

2016), while the target language is always set to
English. Since most of the CCNet dataset is in
English, we filter out the parts that contain only
English text before generating API calls. More
specifically, we only keep those paragraphs which
contain text chunks in a language o...

## A.1
Implementation

Question Answering
We use the Atlas model of
Izacard et al. (2022) finetuned on Natural Ques-
tions (Kwiatkowski et al., 2019) as our question
answering system. For creating C∗we use Atlas-
large, enabling us to efficiently process millions
of API calls; during inference, we use ...

## A.2
Prompts

Calculator
Our calculator is based on a simple
Python script and only supports the operators “+”,
“−”, “∗”, and “/”. It does not return any result
for syntactically invalid equations. For sampling
API calls, we apply heuristic filters to our subset of
CCNet and only process docum...

*[table -- see caption above/below]*

# B
Toolformer Training

*Table 11: Templates used to create DATESET where
a current_date is randomly selected.
For each cur-
rent_date, a random past_date and future_date is gen-
erated and used to fill each template, if relevant. The
federal holidays in the United States (e.g., Thanksgiv-
ing) were used in the templates involving holidays.*

We use up to 25k examples per API. Max sequence
length 1,024. Effective batch size of 128. All mod-
els are trained using DeepSpeed’s ZeRO-3 (Rasley
et al., 2020). We used 8 NVIDIA A100 40GB
GPUs with BF16. Training up to 2k steps, where
we evaluate PPL on a small development set...

# C
Zero-Shot Prompts

## C.1
LAMA and TEMPLAMA

# D
DATESET

For both LAMA and TEMPLAMA, given an input
text x, we use the following prompt: Please
complete the following text so
that it is factually correct:
x.

DATESET is created by first randomly selecting 500
“current dates”. For each current date, another rela-
tively past/future date i...

## C.2
Math Benchmarks

For all math benchmarks, given a context x and
a question q, our prompt is: x q The answer
is.

## C.3
Question Answering

For all question answering datasets, including
DATESET, we simply prefix the question with
Answer the following question:. We
append a question mark if the question does not
already end with one.

## C.4
Multilingual Question Answering

For MLQA, given a context x and a ques-
tion
q,
our
prompt
is:
Your task is

---

<!-- ===== 07_Snell_ScalingTestTimeCompute_ICLR2025.pdf ===== -->

# Scaling LLM Test-Time Compute Optimally can
be More Effective than Scaling Model Parameters

# 1. Introduction

Humans tend to think for longer on difficult problems to reliably improve their decisions [9, 17, 18].
Can we instill a similar capability into today’s large language models (LLMs)? More specifically, given
a challenging input query, can we enable language models to most effectiv...

## Iteratively Revising Answers at Test-time

*[figure -- see caption above/below]*

## Test-time Search Against a PRM Verifier

### Comparing Test-time and Pretraining Compute
in a FLOPs Matched Evauation
rifier

*[figure -- see caption above/below]*

*Figure 1 ∣Summary of our main results. Left: Compute-optimal scaling for iterative self-refinement (i.e., revisions) and search. On
the left, we compare the compute-optimal scaling policy for our PaLM 2-S* revision model against baselines in the revision setting (top) and the
PRM search setting (bottom). We see that in the revisions case, the gap between standard best-of-N (e.g. “parallel”) and compute-optimal
scaling gradually widens, enabling compute-optimal scaling to outperform best-of-N with 4× less test-time compute. Similarly, in the PRM
search setting, we observe significant early improvements over best-of-N from compute-optimal scaling, nearly outperforming best-of-N with 4×
less compute at points. See Sections 5 and 6 for details. Right: Comparing test-time compute and model parameter scaling. We compare
the performance of compute-optimal test-time scaling with PaLM 2-S* against the performance of a ∼14× larger pretrained model without
additional test-time compute (e.g. greedy sampling). We consider the setting where we expect 𝑋tokens of pretraining for both models and 𝑌
tokens of inference. By training a larger model, we effectively multiply the FLOPs requirement for both of these terms. If we were to apply
additional test-time compute with the smaller model, so as to match this larger model’s FLOPs requirement, how would it compare in terms
of accuracy? We see that for the revisions (top) when 𝑌<< 𝑋, test-time compute is often preferable to additional pretraining. However, as
the inference to pretraining token ratio increases, test-time compute remains preferable on easy questions. Whereas on harder questions,
pretraining is preferable in these settings. We also see a similar trend with PRM search (bottom). See Section 7 for more details.*

We are interested in understanding the benefits of scaling up test-time compute. Arguably the simplest
and most well-studied approach for scaling test-time computation is best-of-N sampling: sampling N
outputs in “parallel” from a base LLM and selecting the one that scores the hi...

# 2. A Unified Perspective on Test-Time Computation: Proposer and Verifier

We first unify approaches for using test-time computation and then analyze some representative methods.
First, we view the use of additional test-time compute through the lens of modifying the model’s predicted
distribution adaptively at test-time, conditioned on a given prompt. ...

# 3. How to Scale Test-Time Computation Optimally

Given the unification of various methods, we would now like to understand how to most effectively utilize
test-time computation to improve LM performance on a given prompt. Concretely we wish to answer:

# Problem setup

We are given a prompt and a test-time compute budget within which to solve the problem. Under
the abstraction above, there are different ways to utilize test-time computation. Each of these
methods may be more or less effective depending on the specific problem given. How can we
...

## 3.1. Test-Time Compute-Optimal Scaling Strategy

In general, we would therefore like to select the optimal allocation of our test-time compute budget
for a given problem. To this end, for any given approach of utilizing test-time compute (e.g., revisions
and search against a verifier in this paper, various other methods elsewhe...

## 3.2. Estimating Question Difficulty for Compute-Optimal Scaling

In order to effectively analyze the test-time scaling properties of the different mechanisms discussed in
Section 2 (e.g. the proposal distribution and the verifier), we will prescribe an approximation to this
optimal strategy 𝜃∗
𝑞,𝑦∗(𝑞)(𝑁) as a function of a statistic of a given...

# 4. Experimental Setup

We first outline our experimental setup for conducting this analysis with multiple verifier design choices
and proposal distributions, followed by the analysis results in the subsequent sections.

Datasets. We expect test-time compute to be most helpful when models already have all...

# 5. Scaling Test-Time Compute via Verifiers

In this section we analyze how test-time compute can be scaled by optimizing a verifier, as effectively as
possible. To this end, we study different approaches for performing test-time search with process verifiers
(PRMs) and analyze the test-time compute scaling properties of th...

## 5.1. Training Verifiers Amenable to Search

PRM training. Originally PRM training [22, 42] used human crowd-worker labels. While Lightman
et al. [22] released their PRM training data (i.e., the PRM800k dataset), we found this data to be largely

ineffective for us. We found that it was easy to exploit a PRM trained on this d...

## 5.2. Search Methods Against a PRM

We optimize the PRM at test time via search methods. We study three search approaches that sample
outputs from a few-shot prompted base LLM (see Appendix G). An illustration is shown in Figure 2.

Best-of-N weighted. We sample N answers independently from the base LLM and then sele...

*[figure -- see caption above/below]*

*Figure 2 ∣Comparing different PRM search methods. Left: Best-of-N samples N full answers and then selects the best
answer according to the PRM final score. Center: Beam search samples N candidates at each step, and selects the top M
according to the PRM to continue the search from. Right: lookahead-search extends each step in beam-search to utilize a k-step
lookahead while assessing which steps to retain and continue the search from. Thus lookahead-search needs more compute.*

## 5.3. Analysis Results: Test-Time Scaling for Search with Verifiers

We now present our results comparing various search algorithms and identify a prompt difficulty depen-
dent compute-optimal scaling strategy for search methods.

Comparing search algorithms. We first conduct a sweep over various search settings. In addition to the
standard best-of-...

*[figure -- see caption above/below]*

### Comparing Beam Search and Best-of-N by Difficulty Level

*[figure -- see caption above/below]*

*Figure 3 ∣Left: Comparing different methods for conducting search against PRM verifiers. We see that at low generation
budgets, beam search performs best, but as we scale the budget further the improvements diminish, falling below the best-of-N
baseline. Lookahead-search generally underperforms other methods at the same generation budget. Right: Comparing beam
search and best-of-N binned by difficulty level. The four bars in each difficulty bin correspond to increasing test-time compute
budgets (4, 16, 64, and 256 generations). On the easier problems (bins 1 and 2), beam search shows signs of over-optimization
with higher budgets, whereas best-of-N does not. On the medium difficulty problems (bins 3 and 4), we see beam search
demonstrating consistent improvements over best-of-N.*

To compare search methods as a function of generation budget fairly, we build a protocol for estimating
the cost of each method. We consider a generation to be a sampled answer from the base LLM. For beam
search and best-of-N the generation budget corresponds to the number of bea...

*[figure -- see caption above/below]*

*Figure 4 ∣Comparing compute-optimal test-time compute
allocation against baselines with PRM search. By scaling test
time compute per the notion of question difficulty, we find that
we can nearly outperform PRM best-of-N using up to 4× less
test-time compute (e.g. 16 verses 64 generations). “Compute-
optimal oracle” refers to using oracle difficulty bins derived
from the groundtruth correctness information, and “compute-
optimal predicted” refers to using the PRM’s predictions to
generate difficulty bins. Observe that the curves with either type
of difficulty bins largely overlap with each other.*

# Takeaways for compute-optimal scaling of verifiers

We find that the efficacy of any given verifier search method depends critically on both the compute
budget and the question at hand. Specifically, beam-search is more effective on harder questions
and at lower compute budgets, whereas best-of-N is more effective on easier questi...

# 6. Refining the Proposal Distribution

So far, we studied the test-time compute scaling properties of search against verifiers. Now we turn
to studying the scaling properties of modifying the proposal distribution (Section 2). Concretely, we
enable the model to revise their own answers iteratively, allowing the model ...

*[figure -- see caption above/below]*

*Figure 5 ∣Parallel sampling (e.g., Best-of-N) verses sequential revisions. Left: Parallel sampling generates N answers
independently in parallel, whereas sequential revisions generates each one in sequence conditioned on previous attempts. Right:
In both the sequential and parallel cases, we can use the verifier to determine the best-of-N answers (e.g. by applying best-of-N
weighted). We can also allocate some of our budget to parallel and some to sequential, effectively enabling a combination of the
two sampling strategies. In this case, we use the verifier to first select the best answer within each sequential chain and then
select the best answer accross chains.*

## 6.1. Setup: Training and Using Revision Models

Our procedure for finetuning revision models is similar to [28], though we introduce some crucial
differences. For finetuning, we need trajectories consisting of a sequence of incorrect answers followed
by a correct answer, that we can then run SFT on. Ideally, we want the correc...

*[figure -- see caption above/below]*

*[figure -- see caption above/below]*

*Figure 6 ∣Left: Our revision model’s pass@1 at each revision step. Pass@1 gradually improves after each revision step,
even improving beyond the 4 revision steps that it was trained for. We estimate pass@1 at each step by by averaging over the
performance of 4 revision trajectories of length 64 for each question in the test-set. Right: Sequential vs parallel sampling
from the revision model. Comparing performance when generating N initial answers in parallel from our revision model, verses
generating N revisions sequentially, with the model. When using both the verifier and majority voting to select the answer, we
see that generating answers sequentially with the revision model narrowly outperforms generating them in parallel.*

## 6.2. Analysis Results: Test-Time Scaling with Revisions

We saw previously that proposing answers sequentially outperforms proposing them in parallel. However,
we might expect sequential and parallel sampling to have different properties. Sampling answers in
parallel may act as more of a global search process, that could in principle, ...

*[figure -- see caption above/below]*

*[figure -- see caption above/below]*

*Figure 7 ∣Left: Varying the ratio of the generation budget allocated sequential revisions to verses parallel samples. Each
line represents a fixed generation budget as the ratio is changed. We use the verifier for answer selection. We see that while
increased sequential revisions tends to outperform more parallel compute, at higher generation budgets there is an ideal ratio
that strikes a balance between the two extremes. Right: Varying the sequential to parallel ratio for a generation budget of
128 across difficulty bins. Using verifier-based selection, we see that the easier questions attain the best performance with full
sequential compute. On the harder questions, there is an ideal ratio of sequential to parallel test-time compute.*

*Figure 8 ∣Comparing compute-optimal test-time compute
allocation against the parallel compute baseline with our re-
vision model. By optimally scaling test-time compute according
to question difficulty, we find that we can outperform best-of-N
using up to 4x less test-time compute (e.g. 64 samples verses
256). “Compute-optimal oracle” refers to using the oracle
difficulty bins derived from the ground truth correctness infor-
mation, and “compute optimal predicted” refers to using the
PRM’s predictions to produce model-predicted difficulty bins.*

# Takeaways for compute-optimal scaling by refining the proposal distribution with revisions

We find that there exists a tradeoff between sequential (e.g. revisions) and parallel (e.g. standard
best-of-N) test-time computation, and the ideal ratio of sequential to parallel test-time compute
depends critially on both the compute budget and the specific question at hand. S...

# 7. Putting it Together: Exchanging Pretraining and Test-Time Compute

So far, we saw that utilizing additional test-time computation can enable us to represent more complex
distributions than the one predicted by the base LLM itself, thereby improving performance. We now
posit that this increased flexibility of representing distributions means that...

# Question: Exchanging pretraining and test-time compute

Suppose a model was pre-trained with 𝑋FLOPs. Assume that we plan to run 𝑌FLOPs of inference
with this model. If we want to improve performance by increasing the total FLOPs budget by a
factor of 𝑀(i.e., 𝑀(𝑋+ 𝑌) total FLOPs across both pretraining and inference), should we spend
o...

*[figure -- see caption above/below]*

*Figure 9 ∣Tradeoff between pretraining and test-time compute in a FLOPs-matched evaluation. Each line represents the
performance of scaling test-time compute with our compute-optimal policy in each oracle difficulty bin. We plot the results for
revisions on the left and search on the right. The stars represent the greedy pass@1 performance of a base model pretrained
with ∼14 times more parameters. We plot test-time compute budget on the x-axis, and place the stars at three different
locations along the x-axis, each corresponding to the FLOPs equivalent point of comparison between scaling parameters and
scaling test-time compute for three different inference compute loads (e.g. 𝑅= 𝐷inference
𝐷pretrain ). If the star is below the line, this
implies that it is more effective to use test-time compute than to scale model parameters, and if the star is above the line this
implies that scaling parameters is more effective. We see that on the easy questions or in settings with a lower inference load
(e.g. 𝑅<< 1), test-time compute can generally outperform scaling model parameters. However, on the harder questions or in
settings with a higher inference load (e.g. 𝑅>> 1), pretraining is a more effective way to improve performance.*

# Takeaways for exchanging pretraining and test-time compute

Test-time and pretraining compute are not 1-to-1 “exchangeable”. On easy and medium questions,
which are within a model’s capabilities, or in settings with small inference requirement, test-time
compute can easily cover up for additional pretraining. However, on challenging quest...

# 8. Discussion and Future Work

In this work, we conducted a thorough analysis of the efficacy of different techniques that aim to either
improve search against a verifier or to refine an LLM’s proposal distribution, for scaling test-time compute

for math reasoning. In general, we found that the efficacy of a gi...

# Acknowledgements

We thank Yi Su, Rishabh Agarwal, Yinlam Chow, Aleksandra Faust, Vincent Zhuang, George Tucker,
Hao Liu, Jiayi Pan, Ethan Dyer, Behnam Neyshabur, Xavier Garcia, Yamini Bansal, Lampros Lamprou,
Yuxiao Qu, and Amrith Setlur for their feedback on an earlier version of the paper and d...

# References

# Appendices

# A. Related Work

Language model reasoning. Language model performance on challenging mathematical reasoning tasks
has rapidly improved in recent years [20, 22, 25, 32, 39]. These improvements can be attributed to four
primary factors: 1) running continued pretraining on large corpora of math focu...

# B. Additional Revision Results

We plot additional results for majority selection using out PaLM 2-S* revision model in Figure 10. With
majority selection, we see largely similar trends to those found in Figure 7 for verifier selection.

# C. Unsupervised Difficulty Bins

We compute difficulty bins without oracle ground-truth correctness information by averaging the PRM
final-answer score over 2048 samples on each question, so as to obtain a value estimate corresponding to

*[figure -- see caption above/below]*

*[figure -- see caption above/below]*

*Figure 10 ∣Varying the ratio of generation budget allocated to sequential verses parallel samples, using
majority to select the answer, rather than the verifier. Left: Each line represents a fixed generation budget
as the ratio is changed. We see that similar to the verifier case, in the majority case, there exists an ideal
ratio of sequential to parallel test-time compute at a given budget. Right: Analyzing performance across
difficulty bins, we see that the easier questions are mostly invariant the ratio of sequential to parallel,
whereas on the harder questions there is an ideal ratio of sequential to parallel test-time compute.*

the question. We then bin the value for each question in the test-set into f...

# D. PRM Training Details

We finetune our PRM as a binary classifier, where the model predicts a value between 0 and 1 at each
step in the solution. We train the model with soft values obtained from the monte-carlo rollouts, using
a binary cross entropy loss function (e.g. −(𝑦𝑙𝑜𝑔(ˆ𝑦) + (1 −𝑦)𝑙𝑜𝑔(1 −ˆ𝑦)) w...

*[figure -- see caption above/below]*

*[figure -- see caption above/below]*

*Figure 11 ∣Using our PaLM 2-S* PRM to compute difficulty bins without ground truth correctness
information for revisions. On the left we plot verifier selection and on the right we plot majority selectionl
We see largely similar performance trends with these bins as we do with the ground truth ones in Figures 7
and 10.*

*Figure 12 ∣Using our PaLM 2-S* PRM to compute difficulty bins without ground truth correctness
information for PRM search. We see largely similar performance trends with these bins as we do with the
ground truth ones in Figure 3.*

## Comparing PRM Aggregation Strategies

*[figure -- see caption above/below]*

*Figure 13 ∣We compare different methods of aggregating per-step PRM scores to produce a final score
for the full solution: “min” refers to taking the minimum score accross all steps, “prod” takes the product
of all step correctness probabilities, and “last” just uses the last step score. We see that last performs the
best across all aggregation strategies.*

When generating the samples, the base model is prompted to output answers in newline separated a
step-by-step format, as done in Lightman et al. [22]. We then separate each of the answers into steps
using a simple newline splitting procedure. We include details about our prompt i...

# E. Comparing PRM Aggregation Strategies

We compare different methods of aggregating per-step PRM scores to produce a final score for the full
solution. Specifically we compare: 1) taking the minimum score accross all steps as done in Lightman
et al. [22] (e.g. “min”); 2) taking the product of all step correctness proba...

# F. Comparing PRM and ORM

We trained a PRM and ORM model using the PaLM 2-S* base LM. We see in Figure 14, that the PRM
outperforms the ORM, and the gap between the gap between the PRM and ORM grows with the number

*[figure -- see caption above/below]*

*Figure 14 ∣We compare PRM and ORM models finetuned from PaLM 2-S* in a best-of-N evaluation. We
use the PaLM 2-S* base LM to sample outputs, using a few-shot prompt. We see that the PRM greatly
outperforms the ORM at a larg number of samples.*

of samples used. We use the last step prediction from the PRM to score the answers as descri...

# G. Prompting Details

In order to enable the base model to output answers in a step-by-step format to which a PRM can be
applied, we use a 4-shot prompt consisting of randomly selected correct answer examples from the
PRM800k data released by Lightman et al. [22]. Specifically we use answers from the ...

# H. Revision Model Finetuning Details

For fine-tuning the revision model, we follow the procedure outlined in Section 6.1. We first sample 64
outputs per question. We then filter out all answers which end in an invalid solution. For each correct
answer, we then sample a number uniformly between 0 and 4 indicating how...

# I. Revision Model Selection Criteria

As described in Section 6.1, in order to effective use our revision model we need to deploy a criteria for
selecting the best answer both within a revision trajectory and between multiple parallel trajectories. We
use two approaches: 1) ORM verifier; and 2) majority voting.

For th...

# J. Revision Model Verifier Training

We found that the PRM we finetuned on the PaLM 2-S* base model outputs was not as effective when
applied to the PaLM 2-S* revision model’s outputs (see Figure 15(a)), likely due to distribution shift with
the revision model. We therefore, trained a separate ORM verifier to use wi...

## Revision Model Verifier Verses Base-LM PRM

## Revision Model Verifier With Verse Without History

*[figure -- see caption above/below]*

*[figure -- see caption above/below]*

*Figure 15 ∣Left: we compare the ORM we trained on the revision model’s outputs against the PRM we
trained on the PaLM 2-S* base model’s outputs. We see that when applied to outputs from the revision
model, the ORM adapted to the revision model outperforms the PRM, likely due to distribution shift
with the revision model. Right: we ablate the effect of including previous revisions in the revision model
verifier’s context. We see that including revisions in-context helps the verifier slightly, but both settings
still outperform the parallel baseline.*

the verifier see the revision model’s previous answer attempts when scoring the current answer. All other
experiment details are identical to those used for training the PRM.

Empirically, we find that including the revision history in context improves performance slightly (see
Fig...

# K. ReSTEM Revision Model Experiments

We experimented with further optimizing our PaLM 2-S* revision model by training the model with a
simplified RL algorithm: ReSTEM [35]. Specifically, we generated 64 revision trajectories of maximum
length 5 for each question on the MATH training set. We stopped the revision mode...

*[figure -- see caption above/below]*

*Figure 16 ∣Performance of our ReSTEM optimized revision model as the sequential to parallel ratio
is varied. We use majority voting to select the answer. We see that this optimized revision model
demonstrates substantial performance degradations with additional sequential revisions.*

# L. Revision Model Example Outputs

In Figures 17, 18, 19, 20, 21, 22, and 23, we include select examples of our revision model’s outputs.

# M. PRM Beam Search Example Outputs

In Figures 24, 25, 26, 27, 28, and 29, we include select examples of PRM beam search. We include the
PRM score, between 0 and 1, for each step in the examples.

# (untitled section)

*Figure 17 ∣Revision model example 1. The model calculates the sum at the end incorrectly on the first
two attempts, but on the third attempt it succeeds and gets the answer correct.*

# (untitled section)

# (untitled section)

# (untitled section)

# (untitled section)

Figure 18 ∣Revision model example 2. On the first attempt the model takes the incorrect approach, on
the second attempt it gets closer but then makes a mistake towards the end. On the final attempt it gets
to the correct answer.

# (untitled section)

# (untitled section)

*Figure 19 ∣Revision model example 3. On the first attempt the model makes a mistake with the formatting
of the final answer; it corrects this on the second attempt.*

# (untitled section)

*Figure 20 ∣Revision model example 4. On the first few attempts the model fails the base 10 to base 8
conversion. On the final attempt it makes the correct calculation.*

# (untitled section)

# (untitled section)

# (untitled section)

Figure 21 ∣Revision model example 5. On the first two attempts the model makes an error when
converting euclidean to polar coordinates. On the final attempt it does not make these mistakes.

# (untitled section)

Figure 22 ∣Revision model example 6. On the first two attempts the model makes a mistake when
summing the proper divisors of 284. On the third attempt, it evaluates this sum correctly.

# (untitled section)

Figure 23 ∣Revision model example 7. On the first attempt the model evaluates 1
3 + 2 incorrectly. On the
second attempt it corrects this error.

# Figure 24 ∣PRM beam search example 1.

# Figure 25 ∣PRM beam search example 2.

# (untitled section)

# Figure 26 ∣PRM beam search example 3.

# Figure 27 ∣PRM beam search example 4.

# Figure 28 ∣PRM beam search example 5.

# (untitled section)

# Figure 29 ∣PRM beam search example 6.

---

<!-- ===== 08_Wang_SelfConsistency_ICLR2023.pdf ===== -->

# SELF-CONSISTENCY IMPROVES CHAIN OF THOUGHT
REASONING IN LANGUAGE MODELS

# ABSTRACT

Chain-of-thought prompting combined with pre-trained large language models has
achieved encouraging results on complex reasoning tasks. In this paper, we propose
a new decoding strategy, self-consistency, to replace the naive greedy decoding
used in chain-of-thought prompting. It first samples a diverse set of reasoning paths
instead of only taking the greedy one, and then selects the most consistent answer
by marginalizing out the sampled reasoning paths. Self-consistency leverages the
intuition that a complex reasoning problem typically admits multiple different ways
of thinking leading to its unique correct answer. Our extensive empirical evaluation
shows that self-consistency boosts the performance of chain-of-thought prompting
with a striking margin on a range of popular arithmetic and commonsense reasoning
benchmarks, including GSM8K (+17.9%), SVAMP (+11.0%), AQuA (+12.2%),
StrategyQA (+6.4%) and ARC-challenge (+3.9%).

# 1
INTRODUCTION

Although language models have demonstrated remarkable success across a range of NLP tasks, their
ability to demonstrate reasoning is often seen as a limitation, which cannot be overcome solely by
increasing model scale (Rae et al., 2021; BIG-bench collaboration, 2021, inter alia)...

*[figure -- see caption above/below]*

*Figure 1: The self-consistency method contains three steps: (1) prompt a language model using
chain-of-thought (CoT) prompting; (2) replace the “greedy decode” in CoT prompting by sampling
from the language model’s decoder to generate a diverse set of reasoning paths; and (3) marginalize
out the reasoning paths and aggregate by choosing the most consistent answer in the final answer set.*

# 2
SELF-CONSISTENCY OVER DIVERSE REASONING PATHS

A salient aspect of humanity is that people think differently. It is natural to suppose that in tasks
requiring deliberate thinking, there are likely several ways to attack the problem. We propose that
such a process can be simulated in language models via sampling from the langu...

*[table -- see caption above/below]*

*Table 1: Accuracy comparison of different answer aggregation strategies on PaLM-540B.*

# 3
EXPERIMENTS

We conducted a series of experiments to compare the proposed self-consistency method with existing
approaches on a range of reasoning benchmarks. We find that self-consistency robustly improves
reasoning accuracy for every language model considered, spanning a wide range of model...

# 3.1
EXPERIMENT SETUP

Tasks and datasets.
We evaluate self-consistency on the following reasoning benchmarks.3

Language models and prompts.
We evaluate self-consistency over four transformer-based lan-
guage models with varying scales:

We perform all experiments in the few-shot setting, without training...

# 3.2
MAIN RESULTS

We report the results of self-consistency averaged over 10 runs, where we sampled 40 outputs
independently from the decoder in each run. The baseline we compare to is chain-of-thought
prompting with greedy decoding (Wei et al., 2022), referred to as CoT-prompting, which has been
...

*[table -- see caption above/below]*

*Table 2: Arithmetic reasoning accuracy by self-consistency compared to chain-of-thought prompting
(Wei et al., 2022). The previous SoTA baselines are obtained from: a: Relevance and LCA operation
classifier (Roy & Roth, 2015), b: Lan et al. (2021), c: Amini et al. (2019), d: Pi et al. (2022), e:
GPT-3 175B finetuned with 7.5k examples (Cobbe et al., 2021), g: GPT-3 175B finetuned plus an
additional 175B verifier (Cobbe et al., 2021). The best performance for each task is shown in bold.*

*Table 3: Commonsense and symbolic reasoning accuracy by self-consistency compared to chain-
of-thought prompting (Wei et al., 2022). The previous SoTA baselines are obtained from: a:
DeBERTaV3-large + KEAR (Xu et al., 2021b), b: Chowdhery et al. (2022), c: UnifiedQA-FT
(Khashabi et al., 2020). The best performance for each task is shown in bold.*

*[figure -- see caption above/below]*

*Figure 2: Self-consistency (blue) significantly improves accuracy over CoT-prompting with greedy
decoding (orange) across arithmetic and commonsense reasoning tasks, over LaMDA-137B. Sampling
a higher number of diverse reasoning paths consistently improves reasoning accuracy.*

*[table -- see caption above/below]*

*Table 4: Examples where self-consistency helps repair the errors over greedy decode, on PaLM-540B.
Two sampled reasoning paths that are consistent with the ground truth are shown.*

# 3.3
SELF-CONSISTENCY HELPS WHEN CHAIN-OF-THOUGHT HURTS PERFORMANCE

Ye & Durrett (2022) show that sometimes chain-of-thought prompting could hurt performance
compared to standard prompting in few-shot in-context learning. Here we perform a study using
self-consistency to see if it can help fill in the gap, over a set of common NLP tasks, includin...

*[table -- see caption above/below]*

*Table 5: Compare Standard/CoT prompting with self-consistency on common NLP tasks.*

# 3.4
COMPARE TO OTHER EXISTING APPROACHES

We conduct a set of additional studies and show that self-consistency significantly outperforms
existing methods including sample-and-rank, beam search, and ensemble-based approaches.

Comparison to Sample-and-Rank
One commonly used approach to improve generation quality is
sample-...

*[figure -- see caption above/below]*

*Figure 3: Self-consistency significantly outperforms sample-and-rank with the same # of samples.*

*[table -- see caption above/below]*

*Table 6: Compare self-consistency with beam search decoding on the UL2-20B model.*

*Table 7: Self-consistency outperforms prompt-order and multi-prompt ensembles on LaMDA-137B.*

# 3.5
ADDITIONAL STUDIES

We conducted a number of additional experiments to analyze different aspects of the self-consistency
method, including its robustness to sampling strategies and parameters, and how it works with
imperfect prompts and non-natural-language reasoning paths.

Self-Consistency is Robust...

*[figure -- see caption above/below]*

*[figure -- see caption above/below]*

*Figure 4: GSM8K accuracy. (Left) Self-consistency is robust to various sampling strategies and
parameters. (Right) Self-consistency improves performance across language model scales.*

*[table -- see caption above/below]*

*[figure -- see caption above/below]*

*Figure 5: The consistency is cor-
related with model’s accuracy.*

# 4
RELATED WORK

Reasoning in language models.
Language models are known to struggle in Type 2 tasks, such as
arithmetic, logical and commonsense reasoning (Evans, 2010). Previous work has primarily focused
on specialized approaches for improving reasoning (Andor et al., 2019; Ran et al., 2019; G...

# 5
CONCLUSION AND DISCUSSION

We introduced a simple yet effective method called self-consistency, and observed that it significantly
improves accuracy in a range of arithmetic and commonsense reasoning tasks, across four large
language models with varying scales. Beyond accuracy gains, self-consistency is al...

# REPRODUCIBILITY STATEMENT

In experiments, we included four different language models with varying scales. Two of them are pub-
lic models: UL2 is a completely open-sourced model with model checkpoints available at https://
github.com/google-research/google-research/tree/master/ul2; GPT-3 is
also a public ...

# ETHICS STATEMENT

As we stated in the discussion, language models can sometimes generate nonsensical or non-factual
reasoning paths, so one should use language models’ outputs with extra caution. We deal with
reasoning tasks mostly and the generated rationales are only used for inspecting how a mo...

# REFERENCES

# A
APPENDIX

# A.1
ADDITIONAL EXPERIMENT RESULTS

# A.1.1
ROBUSTNESS TO SAMPLING STRATEGIES AND PARAMETERS

In Figure 6 we ablate the results with respect to different sampling strategies and parameters by
varying T in temperature sampling and k in Top-k sampling, on LaMDA-137B. We show that
self-consistency is robust to various sampling strategies and parameters.

*[figure -- see caption above/below]*

*Figure 6: GSM8K accuracy over LaMDA-137B. Self-consistency works under various sampling
strategies and sampling parameters.*

In Figure 7 and Figure...

*[figure -- see caption above/below]*

*Figure 7: Self-consistency (blue) significantly improves accuracy across various arithmetic and
commonsense reasoning tasks, over LaMDA-137B. Sampling a higher number of diverse reasoning
paths consistently improves reasoning accuracy.*

# A.1.2
ROBUSTNESS TO DIFFERENT SETS OF PROMPTS

In Table 9, we further show that self-consistency is quite robust to different sets of input prompts.
We manually wrote 3 different sets of chain-of-thought as prompts to the model. Across all sets of
prompts, self-consistency yields consistent gains over the original CoT approac...

# A.1.3
COMPARED TO MODEL ENSEMBLES

Additionally, we provide results of directly ensembling the outputs from multiple language models.
The results are shown in Table 10, by greedily decoding sequences from 3 language models and

*[figure -- see caption above/below]*

*Figure 8: Self-consistency (blue) significantly improves accuracy across various arithmetic and
commonsense reasoning tasks, over PaLM-540B. Sampling a higher number of diverse reasoning
paths consistently helps reasoning accuracy.*

*[table -- see caption above/below]*

*Table 9: GSM8K accuracy over PaLM-540B. The results show robustness of self-consistency with
respect to different prompts in the input.*

taking the majority vote (averaged over 10 runs). Note this is a typical ensemble approac...

*[table -- see caption above/below]*

*Table 10: Comparison of GSM8K accuracy over multiple-model ensembles.*

# A.1.4
COMBINING SELF-CONSISTENCY WITH OTHER ENSEMBLING STRATEGIES

Self-consistency is completely compatible with other ensemble strategies, although the gains achieved
by self-consistency are significantly higher than other ensemble strategies (and can “override” the
performance gains achieved by other ensemble strategies). We further performed...

*[table -- see caption above/below]*

*Table 11: Combining self-consistency with other ensembling strategies.*

# A.2
DETAILS ON RESOURCES AND INFERENCE

For all four language models we perform prompting-based inference only. For UL2 we use TPU v3
(2x2 configuration, 4 chips, 8 cores). For GPT-3 models the experiments are done though the public
API.10 For LaMDA-137B we use TPU v3 (8x8 configuration, 64 chips, 128 cores). For PaLM-...

*[table -- see caption above/below]*

*Table 12: Additional examples where self-consistency helps repair the errors over greedy decode on
LaMDA-137B. Two sampled reasoning paths that are consistent with the ground truth are shown.*

# A.3
FULL SETS OF PROMPTS

We list the full details of the prompts used for two newly-introduced datasets, AQUA-RAT (Ling
et al., 2017) and AI2 Reasoning Challenge (ARC) (Clark et al., 2018), where we manually composed
the example chain-of-thought in this paper, in Table 14 and Table 15, respectively.

*Table 14: Few-shot exemplars for AQUA-RAT.*

Q: Jo...

# Table 15: Few-shot exemplars for ARC easy/challenge.

Q: George wants to warm his hands quickly by rubbing them. Which skin surface will produce the most
heat? (a) dry palms. (b) wet palms. (c) palms covered with oil. (d) palms covered with lotion.

A: Dry surfaces will more likely cause more friction via rubbing than other smoother s...

*Table 16: Few-shot exemplars for HotpotQA (closed-book setting).*

# Table 18: Few-shot exemplars for ANLI.

Premise:

"Conceptually cream skimming has two basic dimensions - product and geography."

Based on this premise, can we conclude the hypothesis "Product and geography are what make cream skimming
work." is true?

OPTIONS:

- it is not possible to tell

A: Based on "cream skimming has two ...

# Table 20: Few-shot exemplars for RTE.

Premise:

"No Weapons of Mass Destruction Found in Iraq Yet."

Based on this premise, can we conclude the hypothesis "Weapons of Mass Destruction Found in Iraq." is true?

A: "No Weapons of Mass Destruction Found" contradicts "Weapons of Mass Destruction Found". The answer is
no.

Premis...

*Table 21: Few-shot exemplars for BoolQ (closed-book setting).*

---

<!-- ===== 09_Wei_ChainOfThought_NeurIPS2022.pdf ===== -->

# Chain-of-Thought Prompting Elicits Reasoning
in Large Language Models

# Abstract

We explore how generating a chain of thought—a series of intermediate reasoning
steps—significantly improves the ability of large language models to perform
complex reasoning. In particular, we show how such reasoning abilities emerge
naturally in sufficiently large language models via a simple method called chain-of-
thought prompting, where a few chain of thought demonstrations are provided as
exemplars in prompting.

Experiments on three large language models show that chain-of-thought prompting
improves performance on a range of arithmetic, commonsense, and symbolic
reasoning tasks. The empirical gains can be striking. For instance, prompting a
PaLM 540B with just eight chain-of-thought exemplars achieves state-of-the-art
accuracy on the GSM8K benchmark of math word problems, surpassing even
finetuned GPT-3 with a verifier.

## Chain-of-Thought Prompting

## Standard Prompting

### Model Input

### Model Input

### Model Output

### Model Output

A: The cafeteria had 23 apples originally. They used 
20 to make lunch. So they had 23 - 20 = 3. They 
bought 6 more apples, so they have 3 + 6 = 9. The 
answer is 9.

*[figure -- see caption above/below]*

*Figure 1: Chain-of-thought prompting enables large language models to tackle complex arithmetic,
commonsense, and symbolic reasoning tasks. Chain-of-thought reasoning processes are highlighted.*

# 1
Introduction

The NLP landscape has recently been revolutionized by
language models (Peters et al., 2018; Devlin et al., 2019;
Brown et al., 2020, inter alia). Scaling up the size of lan-
guage models has been shown to confer a range of benefits,
such as improved performance and sample efficie...

*[figure -- see caption above/below]*

*Figure 2:
PaLM 540B uses chain-of-
thought prompting to achieve new state-
of-the-art performance on the GSM8K
benchmark of math word problems.
Finetuned GPT-3 and prior best are from
Cobbe et al. (2021).*

# 2
Chain-of-Thought Prompting

Consider one’s own thought process when solving a complicated reasoning task such as a multi-step
math word problem. It is typical to decompose the problem into intermediate steps and solve each
before giving the final answer: “After Jane gives 2 flowers to her mom she has 10 . ....

# 3
Arithmetic Reasoning

We begin by considering math word problems of the form in Figure 1, which measure the arithmetic
reasoning ability of language models. Though simple for humans, arithmetic reasoning is a task where
language models often struggle (Hendrycks et al., 2021; Patel et al., 2021, inter ...

## 3.1
Experimental Setup

We explore chain-of-thought prompting for various language models on multiple benchmarks.

Benchmarks. We consider the following five math word problem benchmarks: (1) the GSM8K
benchmark of math word problems (Cobbe et al., 2021), (2) the SVAMP dataset of math word
problems with v...

*[figure -- see caption above/below]*

*Figure 3: Examples of ⟨input, chain of thought, output⟩triples for arithmetic, commonsense, and
symbolic reasoning benchmarks. Chains of thought are highlighted. Full prompts in Appendix G.*

## 3.2
Results

The strongest results of chain-of-thought prompting are summarized in Figure 4, with all experimental
outputs for each model collection, model size, and benchmark shown in Table 2 in the Appendix.
There are three key takeaways. First, Figure 4 shows that chain-of-thought promptin...

*[figure -- see caption above/below]*

*Figure 4:
Chain-of-thought prompting enables
large language models to solve challenging math
problems. Notably, chain-of-thought reasoning
is an emergent ability of increasing model scale.
Prior best numbers are from Cobbe et al. (2021)
for GSM8K, Jie et al. (2022) for SVAMP, and Lan
et al. (2021) for MAWPS.*

## 3.3
Ablation Study

The observed benefits of using chain-of-thought prompting raises the natural question of whether the
same performance improvements can be conferred via other types of prompting. Figure 5 shows an
ablation study with three variations of chain of thought described below.

Equation on...

*[figure -- see caption above/below]*

*Figure 5:
Ablation study for dif-
ferent variations of prompting us-
ing LaMDA 137B and PaLM 540B.
Results for other datasets are given
in Appendix Table 6 and Table 7.*

## 3.4
Robustness of Chain of Thought

*[figure -- see caption above/below]*

Sensitivity to exemplars is a key consideration of prompt-
ing approaches—for instance, varying the permutation of
few-shot exemplars can cause the accuracy of GPT-3 on
SST-2 to range from near chance (54.3%) to near state of
the art (93.4%) (Zhao et al., 2021). In this final sub...

*Figure 6: Chain-of-thought prompting
has variance for different prompt exam-
ples (as expected) but outperforms stan-
dard prompting for various annotators as
well as for different exemplars.*

# 4
Commonsense Reasoning

Although chain of thought is particularly suitable for math word problems, the language-based nature
of chain of thought actually makes it applicable to a broad class of commonsense reasoning problems,
which involve reasoning about physical and human interactions under the presum...

*[figure -- see caption above/below]*

*Figure 7:
Chain-of-thought prompting also improves the commonsense reasoning abilities of
language models. The language model shown here is PaLM. Prior best numbers are from the
leaderboards of CSQA (Talmor et al., 2019) and StrategyQA (Geva et al., 2021) (single-model only,
as of May 5, 2022). Additional results using various sizes of LaMDA, GPT-3, and PaLM are shown
in Table 4.*

# 5
Symbolic Reasoning

Standard prompting
Chain-of-thought prompting

Our final experimental evaluation considers symbolic rea-
soning, which is simple for humans but potentially chal-
lenging for language models. We show that chain-of-
thought prompting not only enables language models to
perform symbol...

*[figure -- see caption above/below]*

*Figure 8:
Using chain-of-thought
prompting facilitates generalization to
longer sequences in two symbolic rea-
soning tasks.*

# 6
Discussion

We have explored chain-of-thought prompting as a simple mechanism for eliciting multi-step rea-
soning behavior in large language models. We first saw that chain-of-thought prompting improves
performance by a large margin on arithmetic reasoning, yielding improvements that are mu...

# 7
Related Work

This work is inspired by many research areas, which we detail in an extended related work section
(Appendix C). Here we describe two directions and associated papers that are perhaps most relevant.

The first relevant direction is using intermediate steps to solve reasoning problem...

# 8
Conclusions

We have explored chain-of-thought prompting as a simple and broadly applicable method for enhanc-
ing reasoning in language models. Through experiments on arithmetic, symbolic, and commonsense
reasoning, we find that chain-of-thought reasoning is an emergent property of model sca...

# Acknowledgements

We thank Jacob Devlin, Claire Cui, Andrew Dai, and Ellie Pavlick for providing feedback on the
paper. We thank Jacob Austin, Yuhuai Wu, Henryk Michalewski, Aitor Lewkowycz, Charles Sutton,
and Aakanksha Chowdhery for helpful discussions. We thank Sid Maxwell for notifying us abou...

# References

# Checklist

# A
Frequently Asked Questions

## A.1
Why does increasing model scale improve chain-of-thought prompting?

The finding that successful chain-of-thought reasoning predictably emerges only at certain model
scales is intriguing. Scaling up language models has been shown to confer benefits such as improved
performance and sample efficiency (Kaplan et al., 2020), but chain-of-thought reaso...

*[figure -- see caption above/below]*

*Figure 9: Error analysis of 45 problems that PaLM 62B got incorrect. These errors were categorized
that semantic understanding, one step missing, and other. The other category includes hallucinations,
repetitive outputs, and symbol mapping errors. Scaling PaLM to 540B fixed a substantial portion of
errors in all categories.*

*Figure 10:
Examples of semantic understanding and one-step missing errors that were fixed by
scaling PaLM from 62B to 540B.*

## A.2
What is the role of prompt engineering?

One of the key considerations of prompting is sensitivity to the exact prompt. There is no shortage
of work showing that prompts affect language models in unexpected ways (Min et al., 2022). The
general way that we created chain of thought annotations was by taking eight exemplar...

## A.3
Will chain-of-thought prompting improve performance for my task of interest?

While chain-of-thought prompting is in principle applicable for any text-to-text task, it is more
helpful for some tasks than others. Based on the experiments in this paper, our intuition is that chain
of thought helps the most when three conditions are met: (1) the task is chall...

## A.4
Why is prompting with the equation only not enough for some arithmetic reasoning
datasets?

Prompting with the equation only as an intermediate step does help on many datasets, especially when
the datasets only require a few reasoning steps (SVAMP, ASDiv, MAWPS). For GSM8K, however,
using the equation only did not improve performance substantially. Based on qualitative ...

# B
All Experimental Results

This section contains tables for experimental results for varying models and model sizes, on all
benchmarks, for standard prompting vs. chain-of-thought prompting.

For the arithmetic reasoning benchmarks, some chains of thought (along with the equations produced)
were correct, exc...

*Table 1: Chain of thought prompting outperforms standard prompting for various large language
models on five arithmetic reasoning benchmarks. All metrics are accuracy (%). Ext. calc.: post-hoc
external calculator for arithmetic computations only. Prior best numbers are from the following. a:
Cobbe et al. (2021). b & e: Pi et al. (2022), c: Lan et al. (2021), d: Pi˛ekos et al. (2021).*

*Table 2: Standard prompting versus chain of thought prompting on five arithmetic reasoning bench-
marks. Note that chain of thought prompting is an emergent ability of model scale—it does not
positively impact performance until used with a model of sufficient scale.*

*Table 3: Standard prompting versus chain of thought prompting on the four subsets of the MAWPS
benchmark. The point of stratifying the MAWPS benchmark is to show that performance gains are
minimal on easy one-step or two-step problems where large language models already achieve high
performance (e.g., SingleOp, SingleEq, and AddSub).*

*Table 4: Standard prompting versus chain of thought prompting on five commonsense reasoning
benchmarks. Chain of thought prompting is an emergent ability of model scale—it does not positively
impact performance until used with a model of sufficient scale.*

*Table 5: Standard prompting versus chain of thought prompting enables length generalization to
longer inference examples on two symbolic manipulation tasks.*

*Table 6: Ablation and robustness results for arithmetic reasoning datasets. Chain of thought generally
outperforms ablations by a large amount. “Equation only” performs in between standard prompting
and chain of thought prompting, as it allows for intermediate reasoning steps via equations but does
not leverage natural language. Chain of thought prompting has variance (as expected) when used
with prompts written by different annotators or when using other exemplars, but still outperforms
standard prompting by a large margin. Standard deviation shown is for different order of few-shot
prompting exemplars, with five different random seeds. Results here are shown for LaMDA 137B, as
additional queries for GPT-3 and PaLM are both limited and expensive.*

*Table 7: Ablation and robustness results for four datasets in commonsense and symbolic reasoning.
Chain of thought generally outperforms ablations by a large amount. Chain of thought prompting has
variance (as expected) when used with prompts written by different annotators or when using other
exemplars, but still outperforms standard prompting by a large margin. Standard deviation shown
is for different order of few-shot prompting exemplars, with five different random seeds. Results
here are shown for LaMDA 137B, as additional queries for GPT-3 and PaLM are both limited and
expensive. The exception is that we run SayCan using PaLM here, as the SayCan evaluation set is
only 120 examples and therefore less expensive to run multiple times.*

# C
Extended Related Work

Chain-of-thought prompting is a general approach that is inspired by several prior directions: prompt-
ing, natural language explanations, program synthesis/execution, numeric and logical reasoning, and
intermediate language steps.

## C.1
Prompting

The recent success of large-scale language models has led to growing interest in improving their
capability to perform tasks via prompting (Brown et al. (2020), and see Liu et al. (2021) for a
survey). This paper falls in the category of general prompting approaches, whereby inpu...

## C.2
Natural language explanations

Another closely related direction uses natural language explanations (NLEs), often with the goal of
improving model interpretability (Zhou et al., 2020; Wiegreffe and Marasovi´c, 2021, inter alia). That
line of work typically focuses on natural language inference (Camburu et al.,...

## C.3
Program synthesis and execution

Using intermediate reasoning steps has a long history in program synthesis and execution (Zaremba
and Sutskever, 2014, inter alia). Recent work along in this direction has included a number of
architectural innovations (Cai et al., 2017; Dong et al., 2019; Yan et al., 2020), as w...

## C.4
Numeric and logical reasoning

Numeric and logical reasoning has been a long-studied task in machine learning and natural language
processing (Lev et al., 2004, inter alia). Recent work has also aimed to inject numeric reasoning
abilities in language models in various ways, such as augmenting BERT with a prede...

## C.5
Intermediate language steps

Extensive prior work has shown the benefits of endowing neural networks with the ability to produce
intermediate steps via training or finetuning confers various benefits in a range of scenarios. As
examples, it has been shown that natural language intermediate steps can improve ...

# D
Appendix: Additional Analysis

## D.1
Correct Chain of Thought Analysis

As mentioned in the main text, we analyze 50 chains of thought from LaMDA 137B that led to
correct answers in the GSM8K dataset. Of these 50, only one arrived at the correct answer through
incorrect reasoning (shown in Table 9: “correct by chance”). The other 49 had correct logic...

## D.2
Incorrect Chain of Thought Analysis

We also manually analyze 50 randomly sampled outputs of the model that were incorrect on GSM8K
for LaMDA 137B. There are many ways that a chain of thought can be incorrect, making the design
of error categorization non-trivial. We decided to categorize errors into what changes ar...

## D.3
Additional Robustness Analysis

As the experiments in the main paper use a fixed number of few-shot exemplars (8; as constrained by
the input length of 1024 tokens), we verify that the chain-of-thought prompting is robust to various
numbers of few-shot exemplars. We run experiments for LaMDA 137B, comparing cha...

*[figure -- see caption above/below]*

*Figure 11: The improvement of chain of thought prompting over standard prompting appears robust
to varying the number of few-shot exemplars in the prompt.*

*Table 12: Summary of math word problem benchmarks we use in this paper with examples. N:
number of evaluation examples.*

# E
Additional Details

# Version Control

V5 →V6. Fixed minor typo in Figure 3.

V4 →V5. Added Codex and UL2 results. Small changes to writing and style of paper.

V3 →V4. Fixed typo in Figure 3 and added a couple citations.

V2 →V3. Added GPT-3 results. Added SVAMP and AQuA eval datasets for math. Added SayCan
eval for common...

## E.1
Reproducibility Statement

As our results make use of two sets of large language models that is not publicly available, we take
the following actions to facilitate reproducibility. First, we provide the exact input prompts for all
tasks in Table 20–Table 27 in Appendix G (and emphasize that we do not perfo...

## E.2
Computational Resources

For all three language models we evaluated, we did prompting-based inference only. No finetuning
was done for this paper. For inference on LaMDA 137B we use TPU v3 (8x8 configuration, 64 chips
/ 128 cores), and for inference on PaLM 540B we use TPU v4 (4x4x12 configuration, 192 c...

## E.3
Dataset Details and Licenses

We list the details and licenses for all arithmetic and commonsense datasets used in this paper. The
symbolic reasoning datasets were created synthetically, as described in Section 4.

# Arithmetic reasoning

# Commonsense reasoning

# F
Appendix: Input/Output Examples

Table 13: Examples of correct and incorrect chains of thought produced by LaMDA 137B on the
letter concatenation task.

QUESTION: Take the last letters of the words in “Waldo Schmidt” and concatenate them.

MODEL ANSWER (CORRECT): The last letter of “Waldo” is “o”. The last letter of...

# G
Appendix: Full Prompts

Table 20: Few-shot exemplars for full chain of thought prompt for math word problems. This set of
exemplars was used for all math word problem datasets except AQuA.

# PROMPT FOR MATH WORD PROBLEMS

Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there
will be 21 trees. How many trees did the grove workers plant today?

A: There are 15 trees originally. Then there were 21 trees after some more were planted. So there m...

*Table 21: Few-shot exemplars for full chain of thought prompt for AQuA.*

# PROMPT FOR AQUA ALGEBRAIC WORD PROBLEMS

Q: John found that the average of 15 numbers is 40. If 10 is added to each number then the mean of the numbers
is?

Answer Choices: (a) 50 (b) 45 (c) 65 (d) 78 (e) 64

A: If 10 is added to each number, then the mean of the numbers also increases by 10. So the new mean would be
50. Th...

# PROMPT FOR LAST LETTER CONCATENATION

A: The last letter of "Elon" is "n". The last letter of "Musk" is "k". Concatenating them is "nk". The answer is nk.

Q: Take the last letters of the words in "Larry Page" and concatenate them.

A: The last letter of "Larry" is "y". The last letter of "Page" is "e". Concatenating the...

# PROMPT FOR COIN FLIP

Q: Q: A coin is heads up. Ka flips the coin. Sherrie flips the coin. Is the coin still heads up?

A: The coin was flipped by Ka and Sherrie. So the coin was flipped 2 times, which is an even number. The coin
started heads up, so after an even number of flips, it will still be heads...

# PROMPT FOR CSQA

Q: What do people use to absorb extra ink from a fountain pen? Answer Choices: (a) shirt pocket (b)
calligrapher’s hand (c) inkwell (d) desk drawer (e) blotter

A: The answer must be an item that can absorb ink. Of the above choices, only blotters are used to absorb ink.
So the ans...

# Q: What home entertainment equipment requires cable?

Answer Choices: (a) radio shack (b) substation (c) television (d) cabinet

A: The answer must require cable. Of the above choices, only television requires cable. So the answer is (c).

Q: The fox walked from the city into the forest, what was it looking for? Answer Choices: (a) pret...

# PROMPT FOR STRATEGYQA

Q: Do hamsters provide food for any animals?

A: Hamsters are prey animals. Prey are food for predators. Thus, hamsters provide food for some animals. So
the answer is yes.

Q: Could Brooke Shields succeed at University of Pennsylvania?

A: Brooke Shields went to Princeton University. ...

# PROMPT FOR DATE UNDERSTANDING

Q: 2015 is coming in 36 hours. What is the date one week from today in MM/DD/YYYY?

A: If 2015 is coming in 36 hours, then it is coming in 2 days. 2 days before 01/01/2015 is 12/30/2014, so today
is 12/30/2014. So one week from today will be 01/05/2015. So the answer is 01/05/2015.

# PROMPT FOR SPORTS UNDERSTANDING

Q: Is the following sentence plausible? “Kyle Palmieri was called for slashing.”

A: Kyle Palmieri is a hockey player. Being called for slashing is part of hockey. So the answer is yes.

Q: Is the following sentence plausible? “Joao Moutinho caught the screen pass in the NFC champion...

# PROMPT FOR SAYCAN ROBOT PLANNING

Locations = [counter, table, user, trash, bowl].

Objects = [7up, apple, kettle chips, tea, multigrain chips, coke, lime soda, jalapeno chips, rice chips, orange,
grapefruit soda, pepsi, redbull, energy bar, sponge, water].

The robot can pick up items with pick(object) and put down ...

# H
Appendix: Alternate Annotators for MWP

Table 29: Few-shot exemplars for full chain of thought prompt for math word problems. These
exemplars are the same as in Table 20, except that the chains of thought were written by a different
annotator (“Annotator B” instead of “Annotator A”). Annotators were co-authors and fami...

# PROMPT FOR MATH WORD PROBLEMS

Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there
will be 21 trees. How many trees did the grove workers plant today?

A: There are 21 trees now and there are 15 trees in the beginning, so the workers plant 21 - 15 = 6...

# PROMPT FOR MATH WORD PROBLEMS

Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there
will be 21 trees. How many trees did the grove workers plant today?

A: We start with 15 trees. Later we have 21 trees. The difference must be the number of trees they ...

---

<!-- ===== 10_Yao_TreeOfThoughts_NeurIPS2023.pdf ===== -->

# Tree of Thoughts: Deliberate Problem Solving
with Large Language Models

# Abstract

Language models are increasingly being deployed for general problem solving
across a wide range of tasks, but are still confined to token-level, left-to-right
decision-making processes during inference. This means they can fall short in
tasks that require exploration, strategic lookahead, or where initial decisions play
a pivotal role. To surmount these challenges, we introduce a new framework for
language model inference, “Tree of Thoughts” (ToT), which generalizes over the
popular “Chain of Thought” approach to prompting language models, and enables
exploration over coherent units of text (“thoughts”) that serve as intermediate steps
toward problem solving. ToT allows LMs to perform deliberate decision making
by considering multiple different reasoning paths and self-evaluating choices to
decide the next course of action, as well as looking ahead or backtracking when
necessary to make global choices. Our experiments show that ToT significantly
enhances language models’ problem-solving abilities on three novel tasks requiring
non-trivial planning or search: Game of 24, Creative Writing, and Mini Crosswords.
For instance, in Game of 24, while GPT-4 with chain-of-thought prompting only
solved 4% of tasks, our method achieved a success rate of 74%. Code repo with all
prompts: https://github.com/princeton-nlp/tree-of-thought-llm.

# 1
Introduction

Originally designed to generate text, scaled-up versions of language models (LMs) such as GPT [25,
26, 1, 23] and PaLM [5] have been shown to be increasingly capable of performing an ever wider
range of tasks requiring mathematical, symbolic, commonsense, and knowledge reasoning....

*[figure -- see caption above/below]*

*Figure 1: Schematic illustrating various approaches to problem solving with LLMs. Each rectangle
box represents a thought, which is a coherent language sequence that serves as an intermediate
step toward problem solving. See concrete examples of how thoughts are generated, evaluated, and
searched in Figures 2,4,6.*

# 2
Background

We first formalize some existing methods that use large language models for problem-solving,
which our approach is inspired by and later compared with. We use pθ to denote a pre-trained LM
with parameters θ, and lowercase letters x, y, z, s, · · · to denote a language sequence, i...

# 3
Tree of Thoughts: Deliberate Problem Solving with LM

A genuine problem-solving process involves the repeated use of available informa-
tion to initiate exploration, which discloses, in turn, more information until a way
to attain the solution is finally discovered.—— Newell et al. [21]

Research on human problem-solving suggests that...

# 4
Experiments

We propose three tasks that are hard even when sampling from the state-of-the-art language model,
GPT-4 [23], using standard IO prompting or chain-of-thought (CoT) prompting. We show how

*[table -- see caption above/below]*

*Table 1: Task overview. Input, output, thought examples are in blue.
��������������
����������������
��������������������
���������������������
��
����
��*

deliberate search in trees of thoughts (ToT) produces better results, and more importantly, in...

## 4.1
Game of 24

Game of 24 is a mathematical reasoning challenge, where the goal is to use 4 numbers and basic
arithmetic operations (+-*/) to obtain 24. For example, given input “4 9 10 13”, a solution output
could be “(10 - 4) * (13 - 9) = 24”.
���������������������
���������������������������

*[figure -- see caption above/below]*

*Figure 2: ToT in a game of 24. The LM is prompted for (a) thought generation and (b) valuation.*

*[table -- see caption above/below]*

*[figure -- see caption above/below]*

*[figure -- see caption above/below]*

*Table 2: Game of 24 Results.*

*Figure 3: Game of 24 (a) scale analysis & (b) error analysis.*

## 4.2
Creative writing

Next, we invent a creative writing task where the input is 4 random sentences and the output should
be a coherent passage with 4 paragraphs that end in the 4 input sentences respectively. Such a task is
open-ended and exploratory, and challenges creative thinking as well as high-...

*[figure -- see caption above/below]*

*Figure 4: A step of deliberate search in a randomly picked Creative Writing task. Given the input, the
LM samples 5 different plans, then votes 5 times to decide which plan is best. The majority choice is
used to consequently write the output passage with the same sample-vote procedure.*

*[table -- see caption above/below]*

*CoT > ToT Similar 
Figure 5: Creative Writing results.*

## 4.3
Mini crosswords

In Game of 24 and Creative Writing, ToT is relatively shallow — at most 3 thought steps are needed
to reach the final output. Here we explore 5×5 mini crosswords as a harder search problem involving
natural language. Again, the goal is not just to solve the task, as more general ...

*[figure -- see caption above/below]*

*Figure 6: In Mini Crosswords, (a) how thoughts are proposed and aggregated in a priority queue
for depth-first search (DFS), and (b) how a state is evaluated based on the possibility of filling in
each remaining word clue, and pruned if any remaining clue is deemed not possible to fill by the LM.
Then DFS backtracks to the parent state and explore the next promising thought for clue.*

# 5
Related Work

Planning and decision making. Smart planning and decision making are critical to achieving
predefined goals. As they are trained on vast amount of world knowledge and human examples, LMs
are known to have already absorbed rich commonsense that makes it possible to propose reasona...

# 6
Discussion

Limitations and future directions. Deliberate search such as ToT might not be necessary for many
existing tasks that GPT-4 already excels at (see Appendix B.1), and as an initial step this work only
explores three relatively simple tasks that challenges GPT-4 (see Appendix B.2 fo...

# Broader Impact

ToT is a framework that empowers LMs to more autonomously and intelligently make decisions
and solve problems. While current tasks are limited to reasoning and search problems, future
applications involving interaction with external environments or humans could bring potential da...

# Acknowledgements

SY and KN acknowledge support from an Oracle Collaborative Research award and the National
Science Foundation under Grant No. 2239363. Any opinions, findings, conclusions, or recommenda-
tions expressed in this material are those of the author(s) and do not necessarily reflect th...

# References

# A
Code, Prompts, Trajectories

All code is available at https://github.com/princeton-nlp/tree-of-thought-llm.

All prompts are available at https://github.com/princeton-nlp/tree-of-thought-llm/
tree/master/src/tot/prompts.

Trajectories are available at https://github.com/princeton-nlp/tree-of-thought-llm/
tree/ma...

# B
Additional Experiment Results

Given the motivation of exploring and extending the capability frontier of language models, our
experiments in the main paper have focused on a setup with the state-of-the-art language model
(GPT-4), and three hard tasks invented to challenge it. Here, we report additional experi...

*[table -- see caption above/below]*

## B.1
Extension to new tasks (GSM8k, StrategyQA) with zero-shot ToT

While more common NLP tasks might be too easy for GPT-4 and do not require ToT (which is why
we considered harder new tasks), we believe applying ToT to new tasks could be straightforward.
For example, we implemented a simple and generic zero-shot ToT-BFS similar to creative writ...

## B.2
Extension to new LMs (GPT-3.5)

To understand how ToT works with other LLMs, we also ran GPT-3.5-turbo for Creative Writing
(Table 6) and Game of 24 (Table 5). On both tasks, “ToT > CoT > IO” remains true for GPT-3.5. On
Creative Writing, we find GPT-3.5+ToT outperform GPT-4+IO, and similar to GPT-4+CoT, which
...

## B.3
Cost and efficiency

Running ToT requires significantly more computations than IO or CoT prompting. For example, in
Game of 24 (Table 7 below), solving a problem with ToT requires 5.5k completion tokens, close to
100 CoT trials (6.7k tokens). But the performance of ToT is better than best of 100 inde...

*[table -- see caption above/below]*

*Table 7: Cost analysis on Game of 24.*

*Table 8: Cost analysis on Game of 24.*

---

<!-- ===== 11_Yao_ReAct_ICLR2023.pdf ===== -->

# REACT: SYNERGIZING REASONING AND ACTING IN
LANGUAGE MODELS

# ABSTRACT

While large language models (LLMs) have demonstrated impressive performance
across tasks in language understanding and interactive decision making, their
abilities for reasoning (e.g. chain-of-thought prompting) and acting (e.g. action
plan generation) have primarily been studied as separate topics. In this paper, we
explore the use of LLMs to generate both reasoning traces and task-specific actions
in an interleaved manner, allowing for greater synergy between the two: reasoning
traces help the model induce, track, and update action plans as well as handle
exceptions, while actions allow it to interface with and gather additional information
from external sources such as knowledge bases or environments. We apply our
approach, named ReAct, to a diverse set of language and decision making tasks
and demonstrate its effectiveness over state-of-the-art baselines in addition to
improved human interpretability and trustworthiness. Concretely, on question
answering (HotpotQA) and fact verification (Fever), ReAct overcomes prevalent
issues of hallucination and error propagation in chain-of-thought reasoning by
interacting with a simple Wikipedia API, and generating human-like task-solving
trajectories that are more interpretable than baselines without reasoning traces.
Furthermore, on two interactive decision making benchmarks (ALFWorld and
WebShop), ReAct outperforms imitation and reinforcement learning methods by
an absolute success rate of 34% and 10% respectively, while being prompted with
only one or two in-context examples.

# 1
INTRODUCTION

A unique feature of human intelligence is the ability to seamlessly combine task-oriented actions with
verbal reasoning (or inner speech, Alderson-Day & Fernyhough, 2015), which has been theorized to
play an important role in human cognition for enabling self-regulation or strate...

## ��������������

*[figure -- see caption above/below]*

��������������������������������������������������������
�
���������������������������������������������������
�
��������������������������

*[figure -- see caption above/below]*

## ������������

���������������������������������������������������������
����������������������������������������������������������
��
�����������������������
�������������������������������������������
����������������������������������������������������������������������������
���������������...

*Figure 1: (1) Comparison of 4 prompting methods, (a) Standard, (b) Chain-of-thought (CoT,
Reason Only), (c) Act-only, and (d) ReAct (Reason+Act), solving a HotpotQA (Yang et al., 2018)
question. (2) Comparison of (a) Act-only and (b) ReAct prompting to solve an AlfWorld (Shridhar
et al., 2020b) game. In both domains, we omit in-context examples in the prompt, and only show task
solving trajectories generated by the model (Act, Thought) and the environment (Obs).*

# 2
REAC T: SYNERGIZING REASONING + ACTING

Consider a general setup of an agent interacting with an environment for task solving. At time
step t, an agent receives an observation ot ∈O from the environment and takes an action at ∈A
following some policy π(at|ct), where ct = (o1, a1, · · · , ot−1, at−1, ot) is the context ...

# 3
KNOWLEDGE-INTENSIVE REASONING TASKS

We begin with knowledge-intensive reasoning tasks like multi-hop question answering and fact
verification. As shown in Figure 1(1d), by interacting with a Wikipedia API, ReAct is able to
retrieve information to support reasoning, while also use reasoning to target what to retriev...

# 3.1
SETUP

Domains
We consider two datasets challenging knowledge retrieval and reasoning: (1) Hot-
PotQA (Yang et al., 2018), a multi-hop question answering benchmark that requires reasoning
over two or more Wikipedia passages, and (2) FEVER (Thorne et al., 2018), a fact verification
bench...

# 3.2
METHODS

ReAct Prompting
For HotpotQA and Fever, we randomly select 6 and 3 cases2 from the training
set and manually compose ReAct-format trajectories to use as few-shot exemplars in the prompts.
Similar to Figure 1(d), each trajectory consists of multiple thought-action-observation step...

*[table -- see caption above/below]*

*[figure -- see caption above/below]*

*[figure -- see caption above/below]*

*Table 1: PaLM-540B prompting results on
HotpotQA and Fever.*

*Figure 2: PaLM-540B prompting results with respect to
number of CoT-SC samples used.*

# 3.3
RESULTS AND OBSERVATIONS

ReAct outperforms Act consistently
Table 1 shows HotpotQA and Fever results using PaLM-
540B as the base model with different prompting methods. We note that ReAct is better than Act
on both tasks, demonstrating the value of reasoning to guide acting, especially for synthesizing ...

*[table -- see caption above/below]*

*Table 2: Types of success and failure modes of ReAct and CoT on HotpotQA, as well as their
percentages in randomly selected examples studied by human.*

*[figure -- see caption above/below]*

*Figure 3: Scaling results for prompting and finetuning on HotPotQA with ReAct (ours) and baselines.*

# 4
DECISION MAKING TASKS

We also test ReAct on two language-based interactive decision-making tasks, ALFWorld and
WebShop, both of which feature complex environments that require agents to act over long horizons
with sparse rewards, warranting the need for reasoning to act and explore effectively.

ALFWorl...

*[table -- see caption above/below]*

*Table 4: Score and suc-
cess rate (SR) on Web-
shop. IL/IL+RL taken
from Yao et al. (2022).*

*Table 3: AlfWorld task-specific success rates (%). BUTLER and
BUTLERg results are from Table 4 of Shridhar et al. (2020b). All
methods use greedy decoding, except that BUTLER uses beam search.*

# 5
RELATED WORK

Language model for reasoning
Perhaps the most well-known work of using LLMs for reasoning
is Chain-of-Thought (CoT) (Wei et al., 2022), which reveals the ability of LLMs to formulate their
own “thinking procedure” for problem solving. Several follow-up works have since been perfo...

# 6
CONCLUSION

We have proposed ReAct – a simple yet effective method for synergizing reasoning and acting in
large language models. Through a diverse set of experiments on multi-hop question-answering, fact
checking, and interactive decision-making tasks, we show that ReAct leads to superior p...

# ACKNOWLEDGMENTS

We thank the support and feedback of many people from Google Brain team and Princeton NLP
Group. This work was supported in part by the National Science Foundation under Grant No.
2107048. Any opinions, findings, and conclusions or recommendations expressed in this material are
t...

# REPRODUCIBILITY STATEMENT

Our main experiments are done on PaLM (Chowdhery et al., 2022), which is not an openly accessible
model yet. To increase reproducibility, we have included all used prompts in Appendix C, additional
experiments using GPT-3 (Brown et al., 2020) in Appendix A.1, and associated GPT-3...

# ETHICS STATEMENT

ReAct prompts large language models to generate more human interpretable, diagnosable, and
controllable task-solving trajectories than previous methods. However, hooking up a large language
model with an action space to interact with external environments (e.g. the web, physical ...

# REFERENCES

Alan Baddeley. Working memory. Science, 255(5044):556–559, 1992.

# A
ADDITIONAL RESULTS

# A.1
GPT-3 EXPERIMENTS

*[table -- see caption above/below]*

*Table 5: ReAct prompting results using PaLM-540B vs. GPT-3 (text-davinci-002, greedy decoding).
On HotpotQA, we randomly sample a subset of 500 validation questions. On ALFWorld, we use all
134 unseen validation task instances, and use the best prompt set according to PaLM-540B.*

We run additional GPT-3 (Brown et al., 2020) experiments to confirm ReAct prompting performance
is general across different large language models. As shown in Table 5, GPT-3 (text-davinci-002,
greedy decoding) consistently outperforms PaLM-540B on HotpotQA and ALFWorld, possibly
...

# A.2
REACT OBTAINS UP-TO-DATE KNOWLEDGE ON HOTPOTQA

*[figure -- see caption above/below]*

*Figure 4: Another example HotpotQA question, where the original label is outdated. Only ReAct is
able to obtain the up-to-date answer thanks to real-world web interaction plus reasoning.*

During trajectory inspection, we also find that sometimes ReAct does not agree with dataset labels as
the labels themselves could be outdated. For example, as shown in Figure 4, the question asks about
the size of a hotel, which increased from the HotpotQA construction time. Whil...

# A.3
HUMAN-IN-THE-LOOP BEHAVIOR CORRECTION ON ALFWORLD

We also explore human-in-the-loop interaction with ReAct, to allow a human to inspect and edit
ReAct’s reasoning traces. Figure 5 shows that by simply removing a hallucinating sentence in Act
17 and adding some hints in Act 23, ReAct can be made to change its behavior drastically...

*[figure -- see caption above/below]*

*Figure 5: A human-in-the-loop behavior correction example with ReAct in AlfWorld. (a) ReAct
trajectory fails due to a hallucinating thought (Act 17). (b) By a human simply editing two thoughts
(Act 17, 23), the ReAct trajectory produces desirable reasoning traces and actions and succeeds.*

# B
EXPERIMENT DETAILS

# B.1
HOTPOTQA FINETUNING DETAILS

For all finetuning we use a batch size of 64. On PaLM-8B, we finetune ReAct and Act methods
for 4, 000 steps and Standard and CoT methods for 2, 000 steps. On PaLM-62B, we finetune
ReAct and Act methods for 4, 000 steps and Standard and CoT methods for 1, 000 steps. We
find ReAct...

# B.2
ALFWORLD IM-STYLE DETAILS

For the IM-style ablation, the same expert trajectories used in ReAct are reannotated with dense
external feedback thoughts within these trajectories, that limit ReAct-IM to only think about (1)
decomposing the current goal and (2) the current subgoal that needs to be completed. ...

# C
PROMPTS

# C.1
HOTPOTQA

# Hotpot QA Prompts

## Original

What is the elevation range for the area that the eastern sector of the
Colorado orogeny extends into?
1,800 to 7,000 ft

Question

Answer

Musician and satirist Allie Goertz wrote a song about the "The Simpsons"
character Milhouse, who Matt Groening named after who?
Richard Nixon

Quest...

*[figure -- see caption above/below]*

# Hotpot QA Prompts – Continued from previous page

Action 3
Finish[director, screenwriter, actor]
Question
Which magazine was started first Arthur’s Magazine or First for Women?
Action 1
Search[Arthur’s Magazine]
Observation 1
Arthur’s Magazine (1844-1846) was an American literary periodical published
in Philadelphia in the 19th ...

*[figure -- see caption above/below]*

## ReAct

What is the elevation range for the area that the eastern sector of the
Colorado orogeny extends into?

Question

Thought 1

I need to search Colorado orogeny, find the area that the eastern sector
of the Colorado orogeny extends into, then find the elevation range of the
area.

Search[C...

# Hotpot QA Prompts – Continued from previous page

Observation 4
The High Plains are a subregion of the Great Plains.
From east to west, the
High Plains rise in elevation from around 1,800 to 7,000 ft (550 to 2,130
m).[3]
Thought 5
High Plains rise in elevation from around 1,800 to 7,000 ft, so the answer
is 1,800 to 7,000 ft.
Ac...

*Hotpot QA Prompts – Continued from previous page*

*[table -- see caption above/below]*

# C.2
FEVER

# FEVER Prompts

Determine if there is Observation that SUPPORTS or REFUTES a Claim, or if
there is NOT ENOUGH INFORMATION.

## Original

Claim

Nikolaj Coster-Waldau worked with the Fox Broadcasting Company.
SUPPORTS

Answer

Stranger Things is set in Bloomington, Indiana.
REFUTES

Claim

Answer

Beautiful reached number two on the Billboard Hot 100 in 2003.
NOT ENOUGH INFO

Claim

Answer

*[figure -- see caption above/below]*

Determine if there is Observation that SUP...

*[figure -- see caption above/below]*

## Fever Prompts – Continued from previous page

Claim

Beautiful reached number two on the Billboard Hot 100 in 2003.

The song peaked at number two on the Billboard Hot 100 in the United States,

Thought

but

not sure if it was in 2003.

Answer

NOT ENOUGH INFO

## ReAct

Determine if there is Observation that SUPPORTS or REFUTES a Claim, or if
there is NOT ENOUGH INFORMATION.

Claim

Nikolaj Coster-Waldau worked with the Fox Broadcasting Company.
I need to search Nikolaj Coster-Waldau and find if he has worked with the
Fox Broadcasting Company.

Though...

## Observation 1

Because he "appeared in the 2009 Fox television film Virtuality", he should
have worked with the Fox Broadcasting Company.
Finish[SUPPORTS]

Thought 2

Action 2

Stranger Things is set in Bloomington, Indiana.

Claim

Thought 1

Action 1

Search[Stranger Things]

Stranger Things is an American s...

## Observation 1

The observation says that it is set in a "fictional town of Hawkins,
Indiana", so it is not set in Bloomington.

## Thought 2

Action 2

Finish[REFUTES]

Beautiful reached number two on the Billboard Hot 100 in 2003.

Claim

Thought 1

Billboard Hot 100 in 2003.

## Action 1
Observation 1

Action 1
Search[Beautiful]

Could not find [Beautiful].
Similar:
[’Beautiful’, ’Beautiful, Beautiful’,
’A Beautiful Mind (film)’, ’Beautiful (Christina Aguilera song)’, ’Life Is
Beautiful’].

## Thought 2

Action 2

Search[Beautiful (Christina Aguilera song)]

Observation 2
"Beautiful" is a song recorded by American singer Christina Aguilera for her
fourth studio album, Stripped (2002).

It does not mention Billboard, so I need to look up "Billboard Hot 100" to
find if it reached number ...

## Thought 3

Action 3

Lookup[Billboard Hot 100]

## Observation 3

(Result 1 / 3) The song peaked at number two on the Billboard Hot 100 in the
United States, where it was certified Gold for 500,000 units shipped.
It only says the song peaked at number two on the Billboard Hot 100, but not
if it was in 2003.
I am not sure if this claim is true o...

# C.3
WEBSHOP

*Table 6: Prompts used for the Webshop experiments in Section 4. We use the prompt on the left for
Act, and use the prompt on the right (add reasoning) for ReAct.*

Instruction: i would like a 3 ounce bottle of bright citrus deodorant for sensitive skin, and price lower than 50.00 dollars
Act
ReAct
Action: search
�
3 ounce bright citrus deodorant sensitive skin
�
Observation:
�
Back to Search
�
Page 1 (Total results: 50)
�
Next
�
�
B078GWRC1...

# C.4
ALFWORLD

*Table 7: An Act prompt on the ALFWorld clean task. No thoughts are provided.*

You are in the middle of a room.
Looking quickly around you, you see a cabinet
13, a cabinet 12, a cabinet 11, a cabinet 10, a cabinet 9, a cabinet 8, a cabinet
7, a cabinet 6, a cabinet 5, a cabinet 4, a cabinet 3, a cabinet 2, a cabinet
1, a coffeemachine 1, a countertop 1, a d...

# Table 8: An ReAct prompt on the ALFWorld clean task.

You are in the middle of a room.
Looking quickly around you, you see a cabinet
13, a cabinet 12, a cabinet 11, a cabinet 10, a cabinet 9, a cabinet 8, a cabinet
7, a cabinet 6, a cabinet 5, a cabinet 4, a cabinet 3, a cabinet 2, a cabinet
1, a coffeemachine 1, a countertop 1, a d...

*Table 9: An ReAct-IM prompt on the ALFWorld clean task.*

# D
TRAJECTORIES

# D.1
FEVER TRAJECTORIES

We provide some trajectories produced by ReAct, CoTand Acton several randomly selected exam-
ples from the FEVER dev set below. Search results are omitted to space.

# Example 2491 (gt: REFUTES)

ReAct
Claim:
Bermuda Triangle is a loosely-defined region in the Pacific Ocean.
Thought 1:
I need to search Bermuda Triangle and find if it is in the Pacific Ocean.
Action 1:
Search[Bermuda Triangle]
Observation 1:
Reportedly haunted locations...
Thought 2:
The observation says t...

# Example 5908 (gt: SUPPORTS)

Example 5908 (gt: SUPPORTS)
ReAct
Claim:
Princess Mononoke is a film.
Thought 1:
I need to search Princess Mononoke and find if it is a film.
Action 1:
Search[Princess Mononoke]
Observation 1:
Princess Mononoke ...
Thought 2:
From the observation, it says that Princess Mononoke i...

# Example 1951 (gt: REFUTES)

ReAct
Claim:
Soyuz was part of the American space program.
Thought 1:
I need to search Soyuz and find if it was part of the American space
program.
Action 1:
Search[Soyuz]
Observation 1:
Soyuz is a transliteration...
Thought 2:
It does not mention the American space program, so I...

# Example 3208 (gt: REFUTES)

Example 3208 (gt: REFUTES)
ReAct
Claim:
Reign Over Me is an American film made in 2010.
Thought 1:
I need to search for Reign Over Me and find if it is an American film made
in 2010.
Action 1:
Search[Reign Over Me]
Observation 1:
Reign Over Me ...
Thought 2:
The observation says ...

# D.2
ALFWORLD TRAJECTORIES

We provide an example trajectory of ReAct, Act, and ReAct-IM on the same ALFWorld game,
pick_clean_then_place_in_recep-Knife-None-CounterTop-10/
trial_T20190909_110445_675754.

## D.2.1
REACT TRAJECTORY

ReAct is able to correctly find the knife, clean it, and then place it on a countertop.

*ReAct ALFWorld trajectory*

You are in the middle of a room.
Looking quickly around you, you see a cabinet 6, a
cabinet 5, a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a coffeemachine 1, a
countertop 3, a counterto...

# D.2.2
ACT TRAJECTORY

Act is able to correctly find the knife, but tries to clean it using a sinkbasin before actually going
to the sinkbasin. From there on, the model is unable to recover, and is stuck repeating the same
sequence of commands. With ReAct, an existing thought would have concretely iden...

*Act ALFWorld trajectory*

## D.2.3
REACT-IM TRAJECTORY

ReAct-IM is able to correctly find the knife, but is unable to clean it as well. The LLM then
gets stuck trying to place this knife on a countertop. One possible issue is the incorrect thought
think:
I need to find a clean knife., which seems to trick the LLM in believing
that th...

## ReAct-IM ALFWorld trajectory

You are in the middle of a room.
Looking quickly around you, you see a cabinet 6, a
cabinet 5, a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a coffeemachine 1, a
countertop 3, a countertop 2, a countertop 1, a drawer 3, a drawer 2, a drawer 1, a
fridge 1, a garbagecan 1, a ...

# D.3
WEBSHOP TRAJECTORIES

*Table 10: Example trajectories for Webshop predicted by Act and ReAct. Compared to Act,
ReAct uses reasoning to find products that satisfy all target attributes.*

Instruction: get me a sixteen pack of apple cinnamon freeze dried banana chips, and price lower than 50.00 dollars
Act
ReAct
Action: search
�
sixteen pack apple cinnamon freeze dried banana
chips
�
Observation:
�
Back to Search
�
Page 1 (Total results: 50)
�
Next
�
�
B0061IVFZE
�...

# E
MORE ANALYSIS

# E.1
SUCCESS AND FAILURE MODES ANALYSIS

We provide some examples corresponding to the success and error mode analysis given in Sec. 3.3.
Search results and non-representative steps are omitted to reduce space.

# Success: True positive

## ReAct

Author David Chanoff has collaborated with a U.S. Navy admiral who served as
the ambassador to the United Kingdom under which President?
I need to search David Chanoff and find the U.S. Navy admiral he

Question:

I need to search David Chanoff and find the U.S. Navy admiral he
colla...

## CoT

What was the name of the team that represented an Arizona university in
the conference that covers a broad expanse of the western United States,
including Arizona, California, and New Mexico?

Question:

Thought:

Let’s think step by step.
The name of the team that represented an Arizo...

# Success: False positive

## ReAct

Did the board game San Marco or About Time come out first?

Question:

I need to search San Marco and About Time, and find which came out first.
Search[San Marco]

Thought 1:

Action 1:

San Marco is one of the ...
The was built in venice.

Observation 1:

San Marco is a sestiere of Venice.
I ...

## CoT

Was Vanderbilt University or Emory University founded first?
Let’s think step by step.
Vanderbilt University was founded in 1873.
Emory
University was founded in 1836.
1873 (Vanderbilt University) > 1836 (Emory
University), so Vanderbilt University was founded first.
Emory Univer...

# Failure: Reasoning error

*[table -- see caption above/below]*

# Failure: Search error

## ReAct

What does the goddess associated with the goddess frigg consists of what
tales?

Question:

I need to search goddess frigg and find its associated goddess.
Search[goddess frigg]

Thought 1:

Action 1:

Could not find goddess frigg.
Similar:
[’Frigg’, ’Frigg and Freyja common
origin hypothe...

# Failure: Hallucination

## CoT

What was the first year a scientific journal published by an organization
located in the Bronx was published?

Question:

Let’s think step by step.
The first year a scientific journal published by
an organization located in the Bronx was published was 1916.
1916 (Label:
1909)

Thought:

# Failure: Label ambiguity

## ReAct

What nationality is the man who developed the dictator game?

Question:

I need to search dictator game, find the man who developed it, then find his
nationality.
Search[dictator game]

Thought 1:

Action 1:

Search[dictator game]

The dictator game is a popular experimental instrument in so...

## CoT

Kurt Fuller from Stockton, CA appeared in which American Detective Comedy
Drama television series created by Steve Franks?

Question:

Let’s think step by step.
Kurt Fuller from Stockton, CA appeared in Psych.
Psych is an American Detective Comedy Drama television series created by
S...

---

<!-- ===== 12_Zelikman_STaR_NeurIPS2022.pdf ===== -->

# STaR: Self-Taught Reasoner
Bootstrapping Reasoning With Reasoning

# Abstract

Generating step-by-step "chain-of-thought" rationales improves language model
performance on complex reasoning tasks like mathematics or commonsense
question-answering. However, inducing language model rationale generation cur-
rently requires either constructing massive rationale datasets or sacrificing accuracy
by using only few-shot inference. We propose a technique to iteratively leverage a
small number of rationale examples and a large dataset without rationales, to boot-
strap the ability to perform successively more complex reasoning. This technique,
the "Self-Taught Reasoner" (STaR), relies on a simple loop: generate rationales to
answer many questions, prompted with a few rationale examples; if the generated
answers are wrong, try again to generate a rationale given the correct answer; fine-
tune on all the rationales that ultimately yielded correct answers; repeat. We show
that STaR significantly improves performance on multiple datasets compared to a
model fine-tuned to directly predict final answers, and performs comparably to fine-
tuning a 30× larger state-of-the-art language model on CommensenseQA. Thus,
STaR lets a model improve itself by learning from its own generated reasoning.

# 1
Introduction

Human decision-making is often the result of extended chains of thought [1, 2]. Recent work has
shown that explicit intermediate reasoning (“rationales”) can improve large language model (LLM)
performance as well [3–8]. For example, [5] demonstrated that LLMs explicitly trained t...

*[figure -- see caption above/below]*

*Figure 1: An overview of STaR and a STaR-generated rationale on CommonsenseQA. We indicate
the fine-tuning outer loop with a dashed line. The questions and ground truth answers are expected to
be present in the dataset, while the rationales are generated using STaR.*

# 2
Background and Related Work

In-context Learning
Recently, a collection of works has emerged exploring the capacity for large
language models to perform in-context learning [11, 12]. In essence, in-context learning treats
few-shot learning as a language modeling problem, by showing a few examples in the cont...

# 3
Method

## 3.1
Rationale Generation Bootstrapping (STaR Without Rationalization)

We are given a pretrained LLM M and an initial dataset of problems x with answers y: D =
{(xi, yi)}D
i=1. Our technique starts with a small prompt set P of examples with intermediate ratio-
nales r: P = {(xp
i , rp
i , yp
i )}P
i=1, where P ≪D (e.g. P = 10). Like standard few-sho...

## 3.2
Rationalization

The rationale generation bootstrapping algorithm car-
ries a limitation. Since the model is only trained on
the examples which it answers correctly, improve-
ment ends when the model fails to solve new prob-
lems in the training set. This is fundamentally due
to the fact that the...

# Algorithm 1 STaR

Input M: a pretrained LLM; dataset D = {(xi, yi)}D
i=1 (w/ few-shot prompts)

Algorithm 1 describes the full algorithm, with the parts in blue corresponding to rationalization.
Without those parts, Algorithm 1 corresponds to STaR without rationalization. Figure 1 provides an
overvi...

# 4
Experiments

For our experiments, we focus on arithmetic, commonsense reasoning, and grade school math to
demonstrate STaR’s breadth. In particular, for arithmetic, we follow a setup inspired by [5]. For
commonsense question-answering we follow [13, 6] and use CommonsenseQA (CQA), a widely
us...

## 4.1
Experimental Protocol

We used GPT-J as our base language model, and the fine-tuning script from the GPT-J repository
[26]. We chose GPT-J, a 6B-parameter model, because the checkpoint and fine-tuning code are
publicly available [26], and the model is large enough to generate rationales of non-trivial ...

## 4.2
Datasets

Arithmetic
The arithmetic task is to calculate the sum of two n-
digit integers. We generate the dataset based on the descriptions
in [5] and visualize an example scratchpad in Figure 3. Everything
up to and including “Target:” is given as part of a prompt, and
the model is asked...

*[figure -- see caption above/below]*

*[figure -- see caption above/below]*

*Figure 4: A visualization of the accuracy of n-digit summation with each iteration of STaR with
and without rationalization for arithmetic. Each series corresponds to the accuracy of summing two
n-digit numbers.*

## 4.3
Symbolic Reasoning: Results on Arithmetic

The accuracies of the model across digits 1-5 over each iteration of the outer loop are plotted in
Figure 4. After running STaR for 16 iterations, the overall accuracy is 89.5%. For reference, a
baseline trained on 10,000 examples without rationales for 5,000 steps attains 76.3% ...

*[figure -- see caption above/below]*

*Figure 5: We introduce additional digits to
STaR with rationalization at the 20th iteration.*

*Table 1: We evaluate several baselines, including a few-shot GPT-J evaluation both with and without
scratchpads, a GPT-J baseline finetuned to directly predict the answer, and STaR with and without
rationalization applied to GPT-J. We use CoT to denote non-STaR models outputting rationales, and
Direct to indicate those directly predicting the final answer. Note the final STaR model is trained on
78.2% of the training dataset with rationale generation, and an additional 8.5% from rationalization.*

## 4.4
Natural Language Reasoning: Commonsense Question Answering

The CommonsesenseQA (CQA) setting introduces several new challenges. In the arithmetic task,
an incorrect scratchpad in the reasoning step, and to a lesser degree in the rationalization step, was
extremely likely to result in an incorrect answer. On the other hand, CQA problems a...

*Table 2: We find that STaR substantially improves GSM8K performance over the baselines, despite
training on only 25.0% of the data for the model without rationalization, and 28.7% of the dataset
(with 0.5% from rationalization) for the model with rationalization.*

## 4.5
Mathematical Reasoning in Language: Grade School Math

*[figure -- see caption above/below]*

We again find on GSM8K that STaR substantially
improves performance beyond few-shot with ra-
tionales or training to directly predict the answers
(without rationales), shown in Table 2 and include
the few-shot prompt in Appendix I. We observe
that on this task, the use of rationa...

*Figure 6: A comparison of the number of calcu-
lator steps generated by the model in order to
solve examples in the training set relative to the
number of steps used in the ground truth.*

# 5
Discussion and Challenges

The Impact of Rationalization
An essential question is exactly what role rationalization plays.
Intuitively, rationalization allows a model to reverse-engineer a solution, or provides a heuristic for
identifying whether each step makes the conclusion more likely. This parallels r...

# 6
Conclusion

We present the Self-Taught Reasoner (STaR), which iteratively improves a model’s ability to generate
rationales to solve problems. We few-shot prompt a model to solve many problems in a step-by-step
manner by generating rationales, and then prompt it to rationalize the correct an...

# Acknowledgements

We thank Imanol Schlag for his detailed feedback about this work, as well as Rose E Wang, Markus
Rabe, Aitor Lewkowycz, Rishi Bommasani, Allen Nie, Alex Tamkin, and Qian Huang. We thank
Cem Anil for his very helpful insight that rationale finetuning performance can be improved if...

# References

# Appendix

# A
CommonsenseQA Error Patterns

Throughout our experiments, we came across a variety of interesting failure cases for commonsense
reasoning. Note that all the final answers are correct – however, we take issue with the reasoning
used in order to arrive at those answers.

## A.1
Question Implies Answer

One key failure case was answers in the form of “the answer must be something that is <question
property>. <answer> is <question property>. Therefore, the correct answer is <answer>.” In these
cases, the model fails to explain why the answer that it has chosen satisfies the quest...

## A.2
Begging the Question

A related but stronger version of the previous failure case, while less common, is particularly
uninsightful. Sometimes the model will imply the answer that it has chosen in its question.

Q: The christian rock band was on a large stage, the singer’s foot got caught
on a wire and h...

## A.3
Exercise to the Reader

A rare failure case is when the model finds it unnecessary to justify its answer. For example:

Q: Other than school or from text where can one gain knowledge?
Answer Choices:

A: The answer must be a place where someone can gain knowledge other than
school or from text. The answer i...

## A.4
World State Assertions

Sometimes, the model will assume that it knows something about a subject or a person whose name
was used as a variable. This leads to somewhat comical examples of reasoning. Part of the reason
for this is that generally, there is an expectation that good rationales will leverage ...

## A.5
Red Herrings

Some errors in reasoning corresponded to the model making a statement which, while technically
true, is not useful in demonstrating the claim.

Q: A squirrel ran up to it’s home, where is the small dog likely to be
barking up?
Answer Choices:

Answer Choices:

A: The answer must be a p...

## A.6
Hint Short-cutting

In the experiments where the model was shown some examples of “hints” during training, in order to
prevent it from losing the ability to perform rationalization over time, the model appeared to pick up
on the fact that the final answer would always correspond to the hinted answer...

# B
Modified CQA Prompts

For reference, we include our modified prompts based closely on those in [6].
Q: What do people use to absorb extra ink from a fountain pen?
Answer Choices:
(a) shirt pocket
(b) calligrapher’s hand
(c) inkwell
(d) desk drawer
(e) blotter
A: The answer must be used to absorb extra...

# C
Human-evaluated Test Prompts

We also selected a random sampling of 50 questions which were correctly answered both few-shot
and by a STaR-trained model (without rationalization), as discussed in 4.4. Presented in a random
order, twenty crowdworkers preferred the STaR-generated answers. We reproduce the examp...

# D
Example Rationalizations on CQA

We include a randomly sampled set of rationalizations which the model is able to produce before
fine-tuning. We observe that sometimes, the model constructs an argument roughly of the form “the
answer must have a set of properties. correct answer has those properties. therefore, ...

# E
STaR Without Rationalization (Rationale Generation Bootstrapping)

For convenience, we include the rationale generation bootstrapping algorithm alone here. However, it
is simply Algorithm 1, with the blue components removed.

# Algorithm 2 Rationale Generation Bootstrapping (STaR without rationalization)

Input M: a pretrained LLM; dataset D = {(xi, yi)}D
i=1 (w/ few-shot prompts)

# F
Case Study

A: The answer must be a place where people
go to learn about things. The answer is
university (b).

Answer Choices:

Figure 7: Case study on STaR. We highlight some problems which the model is not initially able
to solve, as well as rationales which are initially correct but improve....

# G
Bias and Faithfulness

It is important to note that STaR is designed to amplify the reasoning that leads to correct solutions
on a given dataset. The implication of this with a dataset like CommonsenseQA is that if biases are
“useful” in solving the dataset then they will be amplified. This is made wor...

# H
Hyperparameters

GPT-J is a 28-layer decoder-only transformer, with an embedding size of 1024, 16 attention heads of
dimension 256, and an FFN hidden layer of size 16384. It was pre-trained on the Pile [35], with a
vocabulary size of 50.4K

In general, unless otherwise stated, we use a batch size o...

# I
GSM8K Few-shot Prompt

We include the following few-shot prompts for GSM8K, based on the examples in [9].

Q: Natalia sold clips to 48 of her friends in April, and then she sold half as many
clips in May. How many clips did Natalia sell altogether in April and May?
A: Natalia sold 48/2 = <<48/2=24>>24 cl...

# J
STaR GSM8K Solutions

We observe some interesting patterns with the GSM8K solutions proposed by the STaR-trained
model. Typically, when the solution takes substantially fewer calculation steps than the ground truth,
it corresponds to an instance where the model accidentally answered the question corre...

*[table -- see caption above/below]*

*Figure 8: An example problem in the training set where STaR derives a significantly simpler solution
than the ground truth.*

---

<!-- ===== 13_Zeng_GLM4.5_arXiv2025.pdf ===== -->

# GLM-4.5: Agentic, Reasoning, and Coding (ARC)
Foundation Models

# GLM-4.5 Team

Zhipu AI & Tsinghua University

(For the complete list of authors, please refer to the Contribution section)

# Abstract

We present GLM-4.5, an open-source Mixture-of-Experts (MoE) large language
model with 355B total parameters and 32B activated parameters, featuring a
hybrid reasoning method that supports both thinking and direct response modes.
Through multi-stage training on 23T tokens and comprehensive post-training
with expert model iteration and reinforcement learning, GLM-4.5 achieves strong
performance across agentic, reasoning, and coding (ARC) tasks, scoring 70.1% on
TAU-Bench, 91.0% on AIME 24, and 64.2% on SWE-bench Verified. With much
fewer parameters than several competitors, GLM-4.5 ranks 3rd overall among
all evaluated models and 2nd on agentic benchmarks. We release both GLM-4.5
(355B parameters) and a compact version, GLM-4.5-Air (106B parameters), to
advance research in reasoning and agentic AI systems. Code, models, and more
information are available at https://github.com/zai-org/GLM-4.5.

*[figure -- see caption above/below]*

*Figure 1: Average performance on agentic, reasoning, and coding (ARC) benchmarks. Overall,
GLM-4.5 achieves a rank of 3rd, with GLM-4.5-Air following at rank 6th. The models listed are
evaluated as of July 28, 2025.*

# 1
Introduction

Large language models (LLMs) are rapidly evolving from general knowledge repositories [6; 37; 50;
33; 23] into general problem-solvers. The ultimate ambition, often associated with Artificial General
Intelligence (AGI), is to create models with human-level cognitive capabilities ...

# 2
Pre-Training

## 2.1
Architecture

In the GLM-4.5 series, we adopt the MoE architecture, which improves the computational efficiency
of both training and inference. We employ loss-free balance routing [40] and sigmoid gates for
MoE layers [23]. Different from DeepSeek-V3 [23] and Kimi K2 [34], we reduce the width ...

*[figure -- see caption above/below]*

*Figure 2: SWE-bench verified scores vs model parameters. Proprietary models are listed as unknown
at the right side.*

*Table 1: Model architecture of GLM-4.5 and GLM-4.5-Air. When counting parameters, for GLM-4.5
and GLM-4.5-Air, we include the parameters of MTP layers but not word embeddings and the output
layer.*

## 2.2
Pre-Training Data

Our pre-training corpus includes documents from webpages, social media, books, papers, and code
repositories. We carefully design the data processing pipelines for different sources.

Web
The majority of our pre-training documents are English and Chinese webpages crawled
from the I...

*[figure -- see caption above/below]*

*Figure 3: Pre-training and mid-training stages for GLM-4.5. We adapt a multi-stage training recipe
and extend the sequence length from 4K to 128K.*

## 2.3
Mid-Training: Boost Reasoning & Agentic Capacity

After pre-training, we add several stages to further boost the model’s performance on important
application areas. Unlike traditional pre-training on large-scale general documents, these training
stages utilize medium-size domain-specific datasets, including instruction data. The...

## 2.4
Hyper-Parameters

We employed the Muon optimizer [21; 24] for all parameters except word embedding, bias, and
weights for RMSNorm. For hyperparameters, we set the Newton-Schulz iteration steps N to 5,
momentum µ to 0.95, and scaled Muon’s update RMS to 0.2. We observed that the Muon optimizer
can ...

# 3
Post-Training: Expert Model Iteration

We divide the post-training process into two distinct stages. In stage 1 (Expert Training), we construct
expert models specializing in three domains: Reasoning, Agent, and General chat. In stage 2 (Unified
Training), we employ self-distillation techniques to integrate multiple ex...

## 3.1
Supervised Fine-Tuning

We perform Supervised Fine-Tuning (SFT) at the beginning of both Stage 1 (Expert Training) and
Stage 2 (Unified Training). In the expert training stage, the primary role of SFT is to provide a cold
start, empowering the model with basic chat, reasoning, and tool-use capabilities,...

*Figure 4: One example of function call template.*

## 3.2
Reasoning RL

Reasoning RL focuses on enhancing a model’s capabilities in domains that demand logical deduction,
structured problem-solving, and verifiable accuracy. This includes critical areas such as mathematics,
code generation, and scientific reasoning. A defining characteristic of these ...

### AIME24 AVG@32: RL with Difficulty-based Curriculum Learning

*[figure -- see caption above/below]*

*Figure 5: Effectiveness of the two-stage difficulty-based curriculum on AIME’24. The blue line
(our method) switches to extremely difficult problems (pass@8=0, pass@512>0) in the second
stage, showing continued improvement. The red line (baseline) continues with moderate-difficulty
problems and plateaus.*

*Figure 6: Single-stage vs. multi-stage RL at 64K context length. The red line (single-stage at
64K) achieves superior performance. The blue line (multi-stage with progressively increasing length)
suffers from an irreversible performance drop in early stages, limiting its final performance.*

scenarios, the lack of reward variance provides no useful gradient signal, severely hindering training
efficiency. To address this challenge, we employ a two-stage difficulty-based curriculum for RL. The
effectiveness of this and other strategies discussed below is validated thro...

*[figure -- see caption above/below]*

*Figure 7: Ablation Studies for Code and Science RL. (Left) Comparison of loss calculation
methods for code RL. The token-weighted mean loss approach achieves faster convergence compared
to the sequence-mean loss baseline, accelerating the training process. (Right) Ablation on data
sources for science RL on the GPQA-Diamond benchmark. Training exclusively on a small set of
high-quality, expert-verified multiple-choice questions yields the best performance, significantly
outperforming training on mixed-quality data.*

## 3.3
Agentic RL

Reinforcement Learning from Human Feedback (RLHF) helps language models follow human
instructions more faithfully. Applying RL to math and programming contests has further uncovered
strong reasoning abilities and favorable scaling behavior on tasks whose outcomes can be objective...

### 3.3.1
Data Collection and Synthesis for Agents

For web-search tasks and open-domain information seeking, we develop a data-synthesis pipeline
that yields demanding question–answer pairs requiring multi-step reasoning across multiple web
sources. This corpus is designed to sharpen GLM’s ability to uncover elusive, interwoven f...

### 3.3.2
Pushing the Limits with Reinforcement Learning and Iterative Self-distillation

We adopt the group-wise policy optimization algorithm for RL training. For each problem x, we
sample K agent traces {y1, . . . , yk} from the previous policy πold, and optimize the model πθ with
respect to the following objective:

where ¯r(x) =
1
k
�k
i=1 r
�
x, yi
�
is the mean r...

## 3.4
General RL

General RL aims to holistically improve the model’s overall performance, remediate potential issues,
and strengthen key capabilities. Central to our methodology is a multi-source feedback system
that synergizes rule-based feedback, human feedback (RLHF), and model-based feedback ...

*[figure -- see caption above/below]*

*Figure 8: Interaction Turns Scaling for BrowseComp.*

### Reward & SysBench Score During Instruction Following RL

*[figure -- see caption above/below]*

*Figure 9: Training curve of Instruction Following RL without other General RL tasks. During
GRPO training, the instruction following performance (SysBench-ISR) improves in step with the
increasing reward. Up to roughly 1,000 training steps, we have not observed clear evidence of reward
hacking.*

Holistic RL
Holistic RL targets broad performance gains across diverse domains. To this end, we
first construct a balanced dataset of roughly 5,000 prompts spanning 7 primary, 33 secondary, and
139 tertiary categories. Reward signals for Holistic RL are derived from both human an...

## 3.5
RL Infrastructure

Our RL infrastructure is built upon Slime1, an open-source framework we developed. The framework
is engineered with several key optimizations to enhance flexibility, efficiency, and extensibility.

Flexible Hybrid Training and Data Generation Architecture
A core feature of our infr...

*[figure -- see caption above/below]*

*Figure 10: Overview of the Slime RL infrastructure. The system consists of three core modules:
Training (Megatron) – handles the main training process, reads data from the Data Buffer, and
synchronizes parameters with the rollout module after training; Rollout (SGLang + Router) –
generates new data, including rewards and verifier outputs, and writes it to the Data Buffer; Data
Buffer – serves as a bridge module that manages prompt initialization, custom data, and rollout
generation strategies.*

# 4
Evaluation

## 4.1
Evaluation of Base Models

We first evaluate the performance of our base model GLM-4.5-Base. Table 2 shows the comparison
results of the last checkpoint of pre-training of our base model. Please note that the base model has
not been trained on instruction data, and the GLM-4.5-Base scores are from our inte...

*Table 2: Comparison among GLM-4.5-Base and other representative open-source base models.*

## 4.2
Evaluation on 12 (ARC) Benchmarks

We further evaluate our full GLM-4.5 models after Post-Training for all the Agentic, Reasoning, and
Coding (ARC) tasks, on 12 benchmarks: MMLU-Pro, AIME 24, MATH-500, SciCode, GPQA, HLE,
LCB (2407-2501), SWE-Bench Verified, Terminal-Bench, TAU-Bench, BFCL V3, BrowseComp.

### 4.2.1
Evaluation of Agentic Abilities

We evaluate the agentic abilities of GLM-4.5 in two aspects: TAU-bench [48] (including retail and
airline domains) and Berkeley Function Call Leaderboard V3 (BFCL V3) [26], which measures the
model’s ability to call user-defined functions to respond to users’ queries. BrowseComp ...

*Table 3: Results on Agentic Benchmarks. TAU represents TAU-bench [48] and BFCL represents
Berkeley Function Calling Leaderboard [26].*

*Figure 11: One example of user prompt we used for TAU-bench.*

### 4.2.2
Evaluation of Reasoning

We evaluate the reasoning abilities of GLM-4.5 and GLM-4.5-Air on seven benchmarks, including
MMLU-Pro [43], AIME 24, MATH 500 [14], SciCode [36], GPQA [30], Humanity’s Last Exam
(HLE) [28], and LiveCodeBench (LCB) [19]2. For the AIME and GPQA benchmarks, we report the
average ac...

*Table 4: Results on Reasoning Benchmarks. HLE represents Humanity’s Last Exam [28] and LCB
represents LiveCodeBench (2407-2501) [19].*

### 4.2.3
Evaluation of Coding

*Table 5: Results on SWE-bench Verified and Terminal-Bench*

To measure GLM-4.5’s ability to complete real-world coding tasks, we evaluate it on two challenging
benchmarks, SWE-bench Verified [20] and Terminal-Bench [35]. SWE-bench measures the model’s
ability to modify an existing codebase to solve a GitHub issue. The Verified subset is a...

### 4.2.4
Evaluation of General Abilities

*Table 6: Results on Commonly Used General Chat Benchmarks*

To evaluate the model’s general abilities, we employed a set of widely-adopted open-source bench-
mark datasets, encompassing knowledge-intensive evaluations MMLU (EM) [14] and SimpleQA (Cor-
rect) [44], and instruction-following assessments IFEval (Prompt Strict) [52], SysBench ...

### 4.2.5
Evaluation of Safety

To systematically assess the safety alignment of our model, we utilized SafetyBench [51], a compre-
hensive benchmark designed to evaluate the safety of large language models. SafetyBench consists
of 11,435 multiple-choice questions covering seven distinct categories of safety co...

## 4.3
Evaluations for Hands-on Experience

Sometimes, a trained LLM may overfit some predefined benchmarks, which makes the evaluated
results not precisely reflect real-world experience. To overcome this challenge and to gauge our
model’s performance in more realistic situations, we have established a comprehensive manual...

*Table 7: Evaluation Results on SafetyBench*

### 4.3.1
Evaluation of General Chat

To test the practical application capabilities of our models, we curated a diverse dataset of real-
scenario user prompts. These prompts span multiple languages and cover a wide range of categories,
including Mathematics, Text Processing, Text Generation, Subjective QA, Objective...

*Table 8: Human Evaluation Scores on English Prompts. Subj. stands for Subjective. Obj stands for
Objective. Text Gen. stands for Text Generation.*

*Table 9: Manual Evaluation Scores on Chinese Prompts*

*Table 10: Manual Evaluation Scores on Other Language Prompts*

### 4.3.2
Evaluation of Coding Agent

*[figure -- see caption above/below]*

*Figure 12: Head-to-head evaluation results between GLM-4.5 and other models on CC-Bench.*

*Figure 13: Average tool calling success rate and token usage per interaction across different models
on CC-Bench.*

Experimental Setup
To evaluate the agentic coding capabilities of GLM-4.5 in real-world scenarios,
we constructed CC-Bench, a benchmark built on the Claude Code5, encompassing 52 carefully
designed programming tasks across diverse software development domains6. We compare GLM-4.5...

### 4.3.3
Evaluation of Logical Reasoning

To rigorously assess the true logical reasoning capabilities of the models and mitigate the risk of
data contamination from common logical questions found online, we constructed a new, challenging
evaluation set. This set comprises novel and complex logical reasoning problems tha...

*Table 11: Expert Evaluation Scores on Novel Logical Reasoning Problems*

## 4.4
Evaluation of Translation

The New Paradigm of Translation
Translation today extends beyond simple text conversion to
encompass a nuanced understanding of evolving internet slang, cultural context, and domain-specific
terminology:

Netizen Lingo: Translating “yyds” accurately requires recognizing it as the a...

*Table 12: Human Scores on Selected Challenging Translation data*

# 5
Conclusion

In this report, we have introduced the GLM-4.5 model series, including GLM-4.5 and GLM-4.5-Air.
Both models adopt the MoE architecture, which improves the computational efficiency compared to
previous GLM models. GLM-4.5 excels at reasoning, coding, and agentic tasks, ranked in 3...

# 6
Contribution

Contributors’ names are listed in alphabetical order by first name. Names marked with an asterisk (*)
indicate individuals who have since left our team.

# Core Contributors

Bin Chen, Chengxing Xie, Cunxiang Wang, Da Yin, Hao Zeng, Jiajie Zhang, Kedong Wang, Lucen
Zhong, Mingdao Liu, Rui Lu, Shulin Cao, Xiaohan Zhang, Xuancheng Huang, Yao Wei, Yean Cheng,
Yifan An, Yilin Niu, Yuanhao Wen, Yushi Bai, Zhengxiao Du, Zihan Wang (汪子涵), Zilin Zhu

# Contributors

Bohan Zhang, Bosi Wen, Bowen Wu, Bowen Xu*, Can Huang, Casey Zhao, Changpeng Cai, Chao
Yu, Chen Li, Chendi Ge, Chenghua Huang, Chenhui Zhang, Chenxi Xu, Chenzheng Zhu, Chuang Li*,
Congfeng Yin, Daoyan Lin, Dayong Yang, Dazhi Jiang, Ding Ai, Erle Zhu, Fei Wang, Gengzheng
Pan, Guo ...

# Tech Leads

Aohan Zeng, Xin Lv, Qinkai Zheng, Zhenyu Hou

# Advisors

Jie Tang, Yuxiao Dong, Juanzi Li, Hongning Wang, Minlie Huang, Bin Xu, Jidong Zhai, Wenguang
Chen

# Acknowledgement

We are grateful for all the support from Beijing, Shanghai, Tianjin, Hangzhou, Zhuhai, and Chengdu.
Special thanks to our customers and community developers.

# References