# comp9417-homework-1--regularized-regression-numerical-solved
**TO GET THIS SOLUTION VISIT:** [COMP9417 Homework 1- Regularized Regression & Numerical Solved](https://mantutor.com/product/comp9417-homework-1-regularized-regression-numerical-solved/)


---

**For Custom/Order Solutions:** **Email:** mantutorcodes@gmail.com  

*We deliver quick, professional, and affordable assignment help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;115893&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;0&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;0&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;0\/5 - (0 votes)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;COMP9417&nbsp;Homework 1- Regularized Regression \u0026amp; Numerical Solved&quot;,&quot;width&quot;:&quot;0&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 0px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            <span class="kksr-muted">Rate this product</span>
    </div>
    </div>
COMP9417 – Machine Learning

Homework 1: Regularized Regression &amp; Numerical

Optimization

<strong>Introduction </strong>In this homework we will explore some algorithms for <em>gradient </em>based optimization. These algorithms have been crucial to the development of machine learning in the last few decades. The most famous example is the backpropagation algorithm used in deep learning, which is in fact just an application of a simple algorithm known as (stochastic) gradient descent. We will first implement gradient descent from scratch on a deterministic problem (no data), and then extend our implementation to solve a real world regression problem.

<strong>Points Allocation </strong>There are a total of 28 marks.

<ul>
<li>Question 1 a): 2 marks</li>
<li>Question 1 b): 1 mark</li>
<li>Question 1 c): 4 marks</li>
<li>Question 1 d): 1 mark</li>
<li>Question 1 e): 1 mark</li>
<li>Question 1 f): 2 marks</li>
<li>Question 1 g): 3 marks</li>
<li>Question 1 h): 3 marks</li>
<li>Question 1 i): 1 mark</li>
<li>Question 1 j): 4 marks</li>
<li>Question 1 k): 5 marks</li>
<li>Question 1 l): 1 mark</li>
</ul>
<h1>What to Submit</h1>
<ul>
<li>A <strong>single PDF </strong>file which contains solutions to each question. For each question, provide your solution in the form of text and requested plots. For some questions you will be requested to provide screen shots of code used to generate your answer — only include these when they are explicitly asked for.</li>
<li><strong>.py file(s) containing all code you used for the project, which should be provided in a separate .zip file. </strong>This code must match the code provided in the report.</li>
</ul>
1

<ul>
<li>You may be deducted points for not following these instructions.</li>
<li>You may be deducted points for poorly presented/formatted work. Please be neat and make your solutions clear. Start each question on a new page if necessary.</li>
<li>You <strong>cannot </strong>submit a Jupyter notebook; this will receive a mark of zero. This does not stop you from developing your code in a notebook and then copying it into a .py file though, or using a tool such as <strong>nbconvert </strong>or similar.</li>
<li>We will set up a Moodle forum for questions about this homework. Please read the existing questions before posting new questions. Please do some basic research online before posting questions. Please only post clarification questions. Any questions deemed to be <em>fishing </em>for answers will be ignored and/or deleted.</li>
<li>Please check Moodle announcements for updates to this spec. It is your responsibility to check for announcements about the spec.</li>
<li>Please complete your homework on your own, do not discuss your solution with other people in the course. General discussion of the problems is fine, but you must write out your own solution and acknowledge if you discussed any of the problems in your submission (including their name(s) and zID).</li>
<li>As usual, we monitor all online forums such as Chegg, StackExchange, etc. Posting homework questions on these site is equivalent to plagiarism and will result in a case of academic misconduct.</li>
<li>You may <strong>not </strong>use SymPy or any other symbolic programming toolkits to answer the derivation questions. This will result in an automatic grade of zero for the relevant question. You must do the derivations manually.</li>
</ul>
<h1>When and Where to Submit</h1>
<ul>
<li>Due date: Week 4, Monday <strong>June 20th</strong>, 2022 by <strong>5pm</strong>. Please note that the forum will not be actively monitored on weekends.</li>
<li>Late submissions will incur a penalty of 5% per day <strong>from the maximum achievable grade</strong>. For example, if you achieve a grade of 80/100 but you submitted 3 days late, then your final grade will be 80 − 3 × 5 = 65. Submissions that are more than 5 days late will receive a mark of zero.</li>
<li>Submission must be done through Moodle, no exceptions.</li>
</ul>
<h1>Question 1. Gradient Based Optimization</h1>
The general framework for a gradient method for finding a minimizer of a function <em>f </em>: R<em><sup>n </sup></em>→ R is defined by

<em>x</em><sup>(<em>k</em>+1) </sup>= <em>x</em><sup>(<em>k</em>) </sup>− <em>α<sub>k</sub></em>∇<em>f</em>(<em>x<sub>k</sub></em>)<em>,&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; k </em>= 0<em>,</em>1<em>,</em>2<em>,…,&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; </em>(1)

where <em>α<sub>k </sub>&gt; </em>0 is known as the step size, or learning rate. Consider the following simple example of√

minimizing the function <em>g</em>(<em>x</em>) = 2 <em>x</em><sup>3 </sup>+ 1. We first note that <em>g</em><sup>0</sup>(<em>x</em>) = 3<em>x</em><sup>2</sup>(<em>x</em><sup>3 </sup>+ 1)<sup>−1<em>/</em>2</sup>. We then need to choose a starting value of <em>x</em>, say <em>x</em><sup>(0) </sup>= 1. Let’s also take the step size to be constant, <em>α<sub>k </sub></em>= <em>α </em>= 0<em>.</em>1. Then we have the following iterations:

<em>x</em><sup>(1) </sup>= <em>x</em><sup>(0) </sup>− 0<em>.</em>1 × 3(<em>x</em><sup>(0)</sup>)<sup>2</sup>((<em>x</em><sup>(0)</sup>)<sup>3 </sup>+ 1)<sup>−1<em>/</em>2 </sup>= 0<em>.</em>7878679656440357 <em>x</em><sup>(2) </sup>= <em>x</em><sup>(1) </sup>− 0<em>.</em>1 × 3(<em>x</em><sup>(1)</sup>)<sup>2</sup>((<em>x</em><sup>(1)</sup>)<sup>3 </sup>+ 1)<sup>−1<em>/</em>2 </sup>= 0<em>.</em>6352617090300827

<em>x</em><sup>(3) </sup>= 0<em>.</em>5272505146487477

…

and this continues until we terminate the algorithm (as a quick exercise for your own benefit, code this up and compare it to the true minimum of the function which is <em>x</em><sub>∗ </sub>= −1). This idea works for functions that have vector valued inputs, which is often the case in machine learning. For example, when we minimize a loss function we do so with respect to a weight vector, <em>β</em>. When we take the stepsize to be constant at each iteration, this algorithm is known as gradient descent. For the entirety of this question, <strong>do not use any existing implementations of gradient methods, doing so will result in an automatic mark of zero for the entire question. </strong>(a) Consider the following optimisation problem:

min <em>f</em>(<em>x</em>)<em>, </em><em>x</em>∈R<em><sup>n</sup></em>

where

<em>,</em>

and where <em>A </em>∈ R<em><sup>m</sup></em><sup>×<em>n</em></sup>, <em>b </em>∈ R<em><sup>m </sup></em>are defined as

<em>&nbsp;,</em>

and <em>γ </em>is a positive constant. Run gradient descent on <em>f </em>using a step size of <em>α </em>= 0<em>.</em>1 and <em>γ </em>= 0<em>.</em>2 and starting point of <em>x</em><sup>(0) </sup>= (1<em>,</em>1<em>,</em>1<em>,</em>1). You will need to terminate the algorithm when the following condition is met: k∇<em>f</em>(<em>x</em><sup>(<em>k</em>)</sup>)k<sub>2 </sub><em>&lt; </em>0<em>.</em>001. In your answer, clearly write down the version of the gradient steps (1) for this problem. Also, print out the first 5 and last 5 values of <em>x</em><sup>(<em>k</em>)</sup>, clearly indicating the value of <em>k</em>, in the form:

<table width="156">
<tbody>
<tr>
<td width="64"><em>k </em>= 0<em>,</em></td>
<td width="92"><em>x</em><sup>(<em>k</em>) </sup>= [1<em>,</em>1<em>,</em>1<em>,</em>1]</td>
</tr>
<tr>
<td width="64"><em>k </em>= 1<em>,</em></td>
<td width="92"><em>x</em>(<em>k</em>) = ···</td>
</tr>
<tr>
<td width="64"><em>k </em>= 2<em>,</em></td>
<td width="92"><em>x</em>(<em>k</em>) = ···</td>
</tr>
</tbody>
</table>
…

<em>What to submit: an equation outlining the explicit gradient update, a print out of the first 5 (</em><em>k </em>= 5 <em>inclusive) and last 5 rows of your iterations. Use the round function to round your numbers to 4 decimal places. Include a screen shot of any code used for this section and a copy of your python code in solutions.py</em>.

<ul>
<li>In the previous part, we used the termination condition k∇<em>f</em>(<em>x</em><sup>(<em>k</em>)</sup>)k<sub>2 </sub><em>&lt; </em>0<em>.</em>001. What do you think this condition means in terms of convergence of the algorithm to a minimizer of <em>f </em>? How would making the right hand side smaller (say 0<em>.</em>0001) instead, change the output of the algorithm? Explain.</li>
</ul>
<em>What to submit: some commentary</em>.

<ul>
<li>In lab 2, we introduced PyTorch and discussed how to use it to perform gradient descent. In this question you will replicate your previous analysis in part (a) but using PyTorch instead. As in part (a), clearly write down the version of the gradient steps (1) for this problem. Also, print out the first 5 and last 5 values of <em>x</em><sup>(<em>k</em>)</sup>, clearly indicating the value of <em>k</em>. You may use the following code as a template if you find it helpful. Note that you may not make any calls to NumPy here.</li>
</ul>
<table width="574">
<tbody>
<tr>
<td width="574">import torch import torch.nn as nn from torch import optim

A = ### b = ### tol = ### gamma = 0.2 alpha = 0.1

class MyModel(nn.Module):

def __init__(self): super().__init__() self.x = ####

def forward(self, ###):

return ###

model = MyModel() optimizer = ### terminationCond = False

k = 0 while not terminationCond:

### compute loss, find gradients, update, check termination cond. etc
</td>
</tr>
</tbody>
</table>
1

2

3

4

5

6

7

8

9

10

11

12

13

14

15

16

17

18

19

20

21

22

23

24

25

<em>What to submit: a print out of the first 5 (</em><em>k </em>= 5 <em>inclusive) and last 5 rows of your iterations. Use the round function to round your numbers to 4 decimal places. Include a screen shot of any code used for this section and a copy of your python code in solutions.py</em>.

In the next few parts, we will use gradient methods explored above to solve a real machine learning problem. Consider the CarSeats data provided in CarSeats.csv. It contains 400 observations with each observation describing child car seats for sale at one of 400 stores. The features in the data set are outlined below:

<ul>
<li>Sales: Unit sales (in thousands) at each location</li>
<li>CompPrice: Price charged by competitor at each location</li>
<li>Income: Local income level (in thousands of dollars)</li>
<li>Advertising: advertising budget (in thousands of dollars)</li>
<li>Population: local population size (in thousands)</li>
<li>Price: price charged by store at each site</li>
<li>ShelveLoc: A categorical variable with Bad, Good and Medium describing the quality of the shelf location of the car seat</li>
<li>Age: Average age of the local population</li>
<li>Education: Education level at each location</li>
<li>Urban A categorical variable with levels No and Yes to describe whether the store is in an urban location or in a rural one</li>
<li>US: A categorical variable with levels No and Yes to describe whether the store is in the US or not.</li>
</ul>
The target variable is Sales. The goal is to learn to predict the amount of Sales as a function of a subset of the above features. We will do so by running Ridge Regression (Ridge) which is defined as follows

<em>β</em>ˆRidge = argmin&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <em>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ,</em>

where <em>β </em>∈ R<em><sup>p</sup></em><em>,X </em>∈ R<em><sup>n</sup></em><sup>×<em>p</em></sup><em>,y </em>∈ R<em><sup>n </sup></em>and <em>φ &gt; </em>0.

<ul>
<li>We first need to preprocess the data. Remove all categorical features. Then use sklearn.preprocessing.StandardScaler to standardize the remaining features. Print out the mean and variance of each of the standardized features. Next, center the target variable (subtract its mean). Finally, create a training set from the first half of the resulting dataset, and a test set from the remaining half and call these objects X train, X test, Y train and Y test. Print out the first and last rows of each of these.</li>
</ul>
<em>What to submit: a print out of the means and variances of features, a print out of the first and last rows of the 4 requested objects, and some commentary. Include a screen shot of any code used for this section and a copy of your python code in solutions.py</em>.

<ul>
<li>It should be obvious that a closed form expression for <em>β</em><sup>ˆ</sup><sub>Ridge </sub> Write down the closed form expression, and compute the exact numerical value on the training dataset with <em>φ </em>= 0<em>.</em>5. <em>What to submit: Your working, and a print out of the value of the ridge solution based on (X train, Y train). Include a screen shot of any code used for this section and a copy of your python code in solutions.py</em>.</li>
</ul>
We will now solve the ridge problem but using numerical techniques. As noted in the lectures, there are a few variants of gradient descent that we will briefly outline here. Recall that in gradient descent our update rule is

<em>β</em>(<em>k</em>+1) = <em>β</em>(<em>k</em>) − <em>α</em><em>k</em>∇<em>L</em>(<em>β</em>(<em>k</em>))<em>,&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; k </em>= 0<em>,</em>1<em>,</em>2<em>,…,</em>

where <em>L</em>(<em>β</em>) is the loss function that we are trying to minimize. In machine learning, it is often the case that the loss function takes the form

<em>,</em>

i.e. the loss is an average of <em>n </em>functions that we have lablled <em>L<sub>i</sub></em>. It then follows that the gradient is also an average of the form

<em>.</em>

We can now define some popular variants of gradient descent .

<ul>
<li>Gradient Descent (GD) (also referred to as batch gradient descent): here we use the full gradient, as in we take the average over all <em>n </em>terms, so our update rule is:</li>
<li>Stochastic Gradient Descent (SGD): instead of considering all <em>n </em>terms, at the <em>k</em>-th step we choose an index <em>i<sub>k </sub></em>randomly from {1<em>,…,n</em>}, and update</li>
</ul>
<em>β</em>(<em>k</em>+1) = <em>β</em>(<em>k</em>) − <em>α</em><em>k</em>∇<em>L</em><em>i</em><em>k</em>(<em>β</em>(<em>k</em>))<em>,&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; k </em>= 0<em>,</em>1<em>,</em>2<em>,….</em>

Here, we are approximating the full gradient ∇<em>L</em>(<em>β</em>) using ∇<em>L<sub>i</sub></em><em><sub>k</sub></em>(<em>β</em>).

<ul>
<li>Mini-Batch Gradient Descent: GD (using all terms) and SGD (using a single term) represents the two possible extremes. In mini-batch GD we choose batches of size 1 <em>&lt; B &lt; n </em>randomly at each step, call their indices {<em>i<sub>k</sub></em><sub>1</sub><em>,i<sub>k</sub></em><sub>2</sub><em>,…,i<sub>k</sub></em><em><sub>B</sub></em>}, and then we update</li>
</ul>
so we are still approximating the full gradient but using more than a single element as is done in SGD.

<ul>
<li>The ridge regression loss is</li>
</ul>
<em>.</em>

Show that we can write

<em>,</em>

and identify the functions <em>L</em><sub>1</sub>(<em>β</em>)<em>,…,L<sub>n</sub></em>(<em>β</em>). Further, compute the gradients ∇<em>L</em><sub>1</sub>(<em>β</em>)<em>,…,</em>∇<em>L<sub>n</sub></em>(<em>β</em>) <em>What to submit: your working</em>.

<ul>
<li>In this question, you will implement (batch) GD from scratch to solve the ridge regression problem. Use an initial estimate <em>β</em><sup>(0) </sup>= 1<em><sub>p </sub></em>(the vector of ones), and <em>φ </em>= 0<em>.</em>5 and run the algorithm for 1000 epochs (an epoch is one pass over the entire data, so a single GD step). Repeat this for the following step sizes:</li>
</ul>
<em>α </em>∈ {0<em>.</em>000001<em>,</em>0<em>.</em>000005<em>,</em>0<em>.</em>00001<em>,</em>0<em>.</em>00005<em>,</em>0<em>.</em>0001<em>,</em>0<em>.</em>0005<em>,</em>0<em>.</em>001<em>,</em>0<em>.</em>005<em>,</em>0<em>.</em>01}

To monitor the performance of the algorithm, we will plot the value

∆(<em>k</em>) = <em>L</em>(<em>β</em>(<em>k</em>)) − <em>L</em>(<em>β</em>ˆ)<em>,</em>

where <em>β</em><sup>ˆ </sup>is the true (closed form) ridge solution derived earlier. Present your results in a 3 × 3 grid plot, with each subplot showing the progression of ∆<sup>(<em>k</em>) </sup>when running GD with a specific step-size. State which step-size you think is best and let <em>β</em><sup>(<em>K</em>) </sup>denote the estimator achieved when running GD with that choice of step size. Report the following:

<ul>
<li>The train MSE:</li>
<li>The test MSE:</li>
</ul>
<em>What to submit: a single plot, the train and test MSE requested. Include a screen shot of any code used for this section and a copy of your python code in solutions.py</em>.

<ul>
<li>We will now implement SGD from scratch to solve the ridge regression problem. Use an initial estimate <em>β</em><sup>(0) </sup>= 1<em><sub>p </sub></em>(the vector of ones) and <em>φ </em>= 0<em>.</em>5 and run the algorithm for 5 epochs (this means a total of 5<em>n </em>updates of <em>β</em>, where <em>n </em>is the size of the training set). Repeat this for the following step sizes:</li>
</ul>
<em>α </em>∈ {0<em>.</em>000001<em>,</em>0<em>.</em>000005<em>,</em>0<em>.</em>00001<em>,</em>0<em>.</em>00005<em>,</em>0<em>.</em>0001<em>,</em>0<em>.</em>0005<em>,</em>0<em>.</em>001<em>,</em>0<em>.</em>006<em>,</em>0<em>.</em>02}

Present an analogous 3 × 3 grid plot as in the previous question. Instead of choosing an index randomly at each step of SGD, we will cycle through the observations in the order they are stored in X train to ensure consistent results. Report the best step-size choice and the corresponding train and test MSEs. In some cases you might observe that the value of ∆<sup>(<em>k</em>) </sup>jumps up and down, and this is not something you would have seen using batch GD. Why do you think this might be happening?

<em>What to submit: a single plot, the train and test MSE requested and some commentary. Include a screen shot of any code used for this section and a copy of your python code in solutions.py</em>.

<ul>
<li>Based on your GD and SGD results, which algorithm do you prefer? When is it a better idea to use GD? When is it a better idea to use SGD?</li>
<li>Note that in GD, SGD and mini-batch GD, we always update the entire <em>p</em>-dimensional vector <em>β </em>at each iteration. An alternative popular approach is to update each of the <em>p </em>parameters individually. To make this idea more clear, we write the ridge loss <em>L</em>(<em>β</em>) as <em>L</em>(<em>β</em><sub>1</sub><em>,β</em><sub>2 </sub><em>…,β<sub>p</sub></em>). We initialize <em>β</em><sup>(0)</sup>, and then solve for <em>k </em>= 1<em>,</em>2<em>,</em>3<em>,…,</em></li>
</ul>
= argmin

= argmin

<em>β</em><sub>2</sub>

…

= argmin<em>.</em>

<em>β<sub>p</sub></em>

Note that each of the minimizations is over a single (1-dimensional) coordinate of <em>β</em>, and also that as as soon as we update, we use the new value when solving the update for and so on. The idea is then to cycle through these coordinate level updates until convergence. In the next two parts we will implement this algorithm from scratch for the Ridge regression problem:

Note that we can write the <em>n </em>× <em>p </em>matrix <em>X </em>= [<em>X</em><sub>1</sub><em>,…,X<sub>p</sub></em>], where <em>X<sub>j </sub></em>is the <em>j</em>-th column of <em>X</em>. Find the solution of the optimization

<em>β</em><sup>ˆ</sup><sub>1 </sub>= argmin<em>L</em>(<em>β</em><sub>1</sub><em>,β</em><sub>2</sub><em>,…,β<sub>p</sub></em>)<em>.</em>

<em>β</em><sub>1</sub>

Based on this, derive similar expressions for <em>β</em><sup>ˆ</sup><em><sub>j </sub></em>for <em>j </em>= 2<em>,</em>3<em>,…,p</em>.

<strong>Hint: </strong>Note the expansion: <em>Xβ </em>= <em>X<sub>j</sub>β<sub>j </sub></em>+ <em>X</em><sub>−<em>j</em></sub><em>β</em><sub>−<em>j</em></sub>, where <em>X</em><sub>−<em>j </em></sub>denotes the matrix <em>X </em>but with the <em>j</em>-th column removed, and similarly <em>β</em><sub>−<em>j </em></sub>is the vector <em>β </em>with the <em>j</em>-th coordinate removed. <em>What to submit: </em>your working out.

<ul>
<li>Implement the algorithm outlined in the previous question on the training dataset. In your implementation, be sure to update the <em>β<sub>j</sub></em>’s in order and use an initial estimate of <em>β</em><sup>(0) </sup>= 1<em><sub>p </sub></em>(th vector of ones), and <em>φ </em>= 0<em>.</em>5. Terminate the algorithm after 10 cycles (one cycle here is <em>p </em>updates, one for each <em>β<sub>j</sub></em>), so you will have a total of 10<em>p </em> Report the train and test MSE of your resulting model. Here we would like to compare the three algorithms: new algorithm to batch GD and SGD from your previous answers with optimally chosen step sizes. Create a plot of <em>k </em>vs. ∆<sup>(<em>k</em>) </sup>as before, but this time plot the progression of the three algorithms. Be sure to use the same colors as indicated here in your plot, and add a legend that labels each series clearly. For your batch GD and SGD include the step-size in the legend. Your x-axis only needs to range from <em>k </em>= 1<em>,…</em>10<em>p</em>. Further, report both train and test MSE for your new algorithm. <em>Note: Some of you may be concerned that we are comparing one step of GD to one step of SGD and the new aglorithm, we will ignore this technicality for the time being. What to submit: a single plot, the train and test MSE requested.</em></li>
<li>In part (d), we standardized the entire data set and then split into train and test sets. In light of this, do you believe that your results in parts (e)-(k) are more reliable, less reliable, or unaffected?</li>
</ul>
Explain. <em>What to submit: your commentary</em>
