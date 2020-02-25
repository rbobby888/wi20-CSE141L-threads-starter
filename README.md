# Using Multiple Cores

In this lab, you will be applying multithreading optimizations to the
perceptron-based ML model you studied in the previous labs.

This lab will be completed on your own.

Check gradescope for due date(s).

## Grading

Your grade for this lab will be based on your completion of the data
collection steps described in this document and the completed worksheet.

| Part                       | value |
|----------------------------|-------|
| Optimizations              | 70%   |
| Worksheet                  | 25%   |
| Reflection                 | 5%    |

The optimizations portions of the grade is based on you successfully
implemented a series of optimizations in Canela, our CNN library.

The optimizations are broken into three tiers.  A total of 100 points are possible.

* Tier 1: Applying multithreading to
   `fc_layer_t::calc_grads()`. (40 points)

* Tier 2: Applying multithreading to
   `fc_layer_t::fix_weights()`.  (40 points)

* Tier 3: Applying additional optimizations to any number of
  functions(from fc_layer, conv_layer, pool_layer, relu_layer) as you
  wish. You are allowed to use loop optimizations that you learnt in
  the previous lab. (20 points)

For Tiers 1 and 2, your score is determined by whether you correctly
implement the optimization specified.  They are all-or-nothing: You
will receive full credit or zero credit depending on whether your
implementation is correct.

For Tier 3, your score will vary depending on how much speedup your
optimizations provide for training the neural network.  Our
measurement show that meeting the requirements Tier 1 and Tier 2 will
provide a speedup of 1.05x relative to the unoptimized code.  The
target speedup for Tier 3 is 6x, and your score cannot be negative.
Your score on Tier 3 is calculated as

```
max(0, (your_speedup-tier2_training_speedup)/(target_training_speedup -
tier2_training_speedup)) * max_points
```

So for this lab that becomes  
```
max(0, (your_speedup-1.05)/(6 - 1.05) * 20)
```

**NOTE** This grading scheme places more emphasis on Tier 3 than the
  previous lab.  On the previous lab, just completing Tier 2 would
  would provide enough speedup on Tier 3 that you get about 94%.  On
  this lab, just completing Tier 2 will get you a 80%.

Your code must pass the regression tests or you'll receive no points
on the lab.

Depending on how things go, we may lower (but will not raise) the
target speedup for Tier 3.  This will only help you.

## The Leader Board

There is a leader board set up for this lab.  It records the speedup of your
code vs the starter code for neural net training.  You can use it to guage your
progress.

For this lab, the leader board does not impact your grade.

## Example Code

The `example` directory contains the image stabilization example from the lab
lecture slides. You shouldn't (and won't need to) use any of the code from
that example for this lab.

## Skills to Learn and Practice

1. Using OpenMP for Multithreading
  
2. Applying loop reordering.

2. Applying loop tiling.

3. Testing code.

4. Quantifying the benefits of optimizations

5. Interpreting performance counters

## Software You Will Need

1. A computer with Docker installed (either the cloud docker container
via ssh, or your own laptop).  See the intro lab for details.

2. The lab for the github classroom assignment for this lab.  Find the
link on the course home page: https://github.com/CSE141pp/Home/.

3. A PDF annotator/editor to fill out `worksheet.pdf`.  You'll
submit this via a *a separate assignment* in Gradescope.  We *will
not* look at the version in your repo.

## Tasks to Perform

### Inspect The Code

There are three source files in the lab, but you'll only be editting one:

1.  `main.cpp` -- The driver code is in `main()` (mostly data
collection and command line parsing).  Also, code for creating,
training, and running ML models. 

2.  `opt_cnn.hpp` -- A soon-to-be-optimized (by you) version of two CNN primitives.  

There is a `code.cpp` but you won't use it in this lab.

You will only be editing `opt_cnn.hpp`.  Changes to the `main.cpp`
will have no effect on the autograder's output.

The basic flow is like this:

* Execution starts in `main.cpp` which loads the input dataset.

* `main.cpp` executes the "canary" to verify the machine is performing properly.

* It measures the performance of your implementation of
  `fc_layer_t::calc_grads()` for Tier 1 grading.

* It measures the performance of your implementation of
  `fc_layer_t::fix_weights()` for Tier 2 grading.

* It measures the performance of neural net training using your
  optimized functions for Tier 3 grading.


You'll find five skeleton classes in `opt_cnn.hpp`.  They inherit from
the corresponding classes in Canela.  Right now, they have no code, so
using them is the same as using original Canela classes.

To optimize them, you should copy the functions from Canela you want
to optimize into these classes.  Any changes you make will effect the
performance correctness of the code in `main.cpp`.

### Test Locally

Like last time, get started by checking out the code and checking it locally with 

```
runlab --devel
```

The code will run for a while.  On our machine, the starter lab runs
for about 140s.  Your local machine may be slower or faster.

You'll get a few files:

1. `regression.out` has the report from the regression suite.

2. `benchmark.csv` is the csv file used to measure performance.
`CMD_LINE_ARGS` has no effect.

3. `code.csv` is similar to `benchmark.csv` but `CMD_LINE_ARGS` has its
normal effect.

4. `code.gprof` and `benchmark.gprof` are not here now, but if you set
`GPROF=yes` they will appear.

You can submit it to the autograder for good measure, if you want.

### Command Line Options

Your executable takes a few useful command line options we haven't discussed:

* `--scale` this sets the input size for the input data set.  The bigger the
  scale, the more inputs we run through the model.  This only affects the
  execution of `train_model`.
  
* `--reps` how many times to run the individual functions for Tier 1 and Tier 2.

### Read the Source

You need to get acquainted with the code you'll be optimizing.  The
slides from the lab lecture are an important resource here, especially
for memory layout of `tensor_t` and how the functions in `fc_layer_t`
work.

The baseline version of Canela is in your docker repo in
`/course/CSE141pp-SimpleCNN/CNN`.  You should read through the
commented code in these files (some of these are familiar from prior
labs.  Nothing significant has changed in those files):

* `tensor_t.hpp`

* `types.hpp`

* `layer_t.hpp`

* `fc_layer_t.hpp`

* `conv_layer_t.hpp`

* `pool_layer_t.hpp`

* `relu_layer_t.hpp`

In each of these files there's some code near the top that has lots
comments.  This is the code you should focus on.  There's a lot more
code down below, but it's all utility functions and debugging
support. It's not important to the operation of the library or your
optimizations.

The point is not deeply understand the code at this point.  Rather,
it's to become acquainted with where the code is.  This will make it
easier to answer questions about the code that you have later.

### Tier 1: Optimizing fc_layer_t::calc_grads()

To get you started, we will walk you through the process for optimizing one
function: `fc_layer_t::calc_grads()`.  Please see the slides in the lab repo for
more details.  They contain detailed description of how the code works.

Unlike the previous labs, the code for the baseline implementation lives in
`opt_cnn.hpp`.

Run `runlab --no-validate`.
It should finish and in the output, you'll see

```
[  PASSED  ] 21 tests.
```

Which means that your implementation matches the result of the
baseline (which is no surprise because you have not edited baseline).

These tests are your best friend, since they provide a quick and easy
way of telling whether your code is correct.  `runlab` runs the tests
every time, and if you the last line shows any failures, you should
look at `regressions.out` for a full report.

**Note** Regressions are always built without optimizations (`-O0`) to make
them debuggable.

**Note** You will want to save the output of benchmark.csv from this run as the single threaded time for the worksheet.

#### nn Loop

First, make a copy of the baseline implementation of `fc_layer_t::calc_grads()`, you'll need it later. You can do this by renaming the function, using define, or just commenting it out. Note, that you just want to create a copy of the baseline and save it somewhere. Make sure this is not the version of the code you run.

Then change OMP_NUM_THREADS to 2 in config.env

Modify the code to add multithreading to the `nn` loop and run it again. You can do this by adding `#pragma omp prallel for` on the line before the `nn` for loop. When your code finishes running, you will notice that you failed multiple regression tests. This is because by parallelizing the `nn` loop, multiple threads attempt to write to the same location in `grads_out`.

We will fix this in two stages:

**Stage 1** Add a local tensor the same size as `grads_out` at the top of the `nn` for loop. You can see an example of this in `exmaple/stabilize.cpp` line 338. The example creates a tensor of type double with the same size as `output`. You will do the same except the tensor size will be the same size as `grads_out`. Do not forget to clear it just like the example does on line 339. Then, in the inner most loop (the `i` loop), change `grads_out` to be your new local tensor that each thread will create.

This enables each thread to accumulate their results locally. Thereby eliminating the race condition causing errors. However, we now need to combine the results from each thread.

**Stage 2** At the bottom of the `nn` for loop, add a critical section and create two nested for loops to loop through `out.size.b` and `grads_out.size.x`. Notice that we don't loop through `out.size.x` as well. This is because we only need to accoumilate the results into `grads_out` and `n` is not used to index into `grads_out`. 

Inside the nested for loop you just created, accumulate the results of each thread (stored in their local tensors you creaded in stage 1) into `grads_out`. This will look very similar to `exmaple/stabilize.cpp` lines 361 - 369. 

Once you have made your changes, run the code locally and verify that you pass all 21 regression tests. If you do not pass, refer back to the lecture slides, discussion slides, example in `exmaple/stabilize.cpp` lines 330 - 372, and help from the staff during office hours or lab hours. Once you have verified that your code is correct and passes the regression tests, submit to the autograder. You will want to save the resulting benchmark.csv file for the worksheet.

#### b Loop

First, save your implementation of multithreading the `nn` loop the same way you saved the baseline. You may need it later. Then revert to the baseline implementation you saved earlier.

Modify the code to add multithreading to the `b` loop and run it again. You can do this by adding `#pragma omp prallel for` on the line before the `b` for loop. You will notice that you passed the regressions test! There is no need to fix any race condition here as each thread is accumulating its result into a different address of grads_out.

Once you have made your changes, run the code locally and verify that you pass all 21 regression tests. If you do not pass, refer back to the lecture slides, discussion slides, example in `exmaple/stabilize.cpp` lines 330 - 372, and help from the staff during office hours or lab hours. Once you have verified that your code is correct and passes the regression tests, submit to the autograder. You will want to save the resulting benchmark.csv file for the worksheet.

#### n Loop

First, save your implementation of multithreading the `b` loop the same way you saved the baseline. You may need it later. Then revert to the baseline implementation you saved earlier.

Modify the code to add multithreading to the `n` loop. You can do this by adding `#pragma omp prallel for` on the line before the `n` for loop. You'll also have to additionally modify the loop condition as it is too complext for openmp. Create an int called `minn` and set it to the minimum between `nn` + `BLOCK_SIZE` and `out.size.x`. Then change your loop condition to `n < minn`. Once you make these changes, run the code again. When your code finishes running, you will notice that you failed multiple regression tests. This is because by parallelizing the `n` loop, multiple threads attempt to write to the same location in `grads_out`.

We will fix this in two stages:

**Stage 1** Add a local tensor the same size as `grads_out` at the top of the `n` for loop. You can see an example of this in `exmaple/stabilize.cpp` line 338. The example creates a tensor of type double with the same size as `output`. You will do the same except the tensor size will be the same size as `grads_out`. Do not forget to clear it just like the example does on line 339. Then, in the inner most loop (the `i` loop), change `grads_out` to be your new local tensor that each thread will create.

This enables each thread to accumulate their results locally. Thereby eliminating the race condition causing errors. However, we now need to combine the results from each thread.

**Stage 2** At the bottom of the `n` for loop, add a critical section and create a for loops to loop through `grads_out.size.x`. Notice that we don't loop through `out.size.x` or `out.size.b` as well. We exclude `out.size.x` because we only need to accoumilate the results into `grads_out` and `n` is not used to index into `grads_out`. We exclude `out.size.b` because we only need to accumilate the result that the threads were individually working on, and they were all already working on the same `b` because we multithreaded the `n` loop, which is inside the `b` loop. 

Inside the for loop you just created, accumulate the results of each thread (stored in their local tensors you creaded in stage 1) into `grads_out`. This will look very similar to `exmaple/stabilize.cpp` lines 361 - 369. 

Once you have made your changes, run the code locally and verify that you pass all 21 regression tests. If you do not pass, refer back to the lecture slides, discussion slides, example in `exmaple/stabilize.cpp` lines 330 - 372, and help from the staff during office hours or lab hours. Once you have verified that your code is correct and passes the regression tests, submit to the autograder. You will want to save the resulting benchmark.csv file for the worksheet.

#### i Loop

First, save your implementation of multithreading the `n` loop the same way you saved the baseline. You may need it later. Then revert to the baseline implementation you saved earlier.

Modify the code to add multithreading to the `b` loop and run it again. You can do this by adding `#pragma omp prallel for` on the line before the `b` for loop. You will notice that you passed the regressions test! There is no need to fix any race condition here as each thread is accumulating its result into a different address of grads_out.

Once you have made your changes, run the code locally and verify that you pass all 21 regression tests. If you do not pass, refer back to the lecture slides, discussion slides, example in `exmaple/stabilize.cpp` lines 330 - 372, and help from the staff during office hours or lab hours. Once you have verified that your code is correct and passes the regression tests, submit to the autograder. You will want to save the resulting benchmark.csv file for the worksheet.

#### Multiple threads

Now that we found the best loop to multithread, try changing
`OMP_NUM_THREADS` to 4 then 6 in `config.env`. Run the code locally
and verify that you pass all 21 regression tests each time you change
`OMP_NUM_THREADS`. When you know it passes the tests, submit to the
autograder. You will want to save the resulting benchmark.csv files
for the worksheet.

Now that we have the best loop multithreaded and know the best number
of threads to use, submit to the autograder.  If you've done
everything correctly, your code should pass Tier 1. The precise target
for the speedup is listed in gradescope output.

### Tier 2: Optimizing fc_layer_t::fix_weights() 

For Tier 2, you need apply `parallel for` optimization to
`fix_weights()` function of the fully connected layer.

First, copy the function from
`/course/CSE141pp-SimpleCNN/CNN/fc_layer_t.hpp` into your
`opt_cnn.hpp`.  Make it a method of the `opt_fc_layer` class.

Similar to Tier 1, you will have to apply multithreading, one loop at
the time, on `b`, `n`, and `i` loops.  Make a note of the speedups
achieved in each case for the worksheet.

If you do that successfully, your code should pass Tier 2.  The
precise target for the speedup is listed in gradescope output.

### Tier 3:  Other optimizations

Go forth and optimize!

There are more opportunities to apply multithreading, loop reordering,
and tiling across `activate()`, `calc_grads()`, and `fix_weights()`
functions from `fc_layer_t`, `conv_layer_t`, `pool_layer_t`, and
`relu_layer_t`. You can apply whatever optimizations you want with the
following restrictions:

1. You can't modify `main.cpp`

2. No explicit vectorization (that's a later lab).

3. Your code must pass all the regression tests.

4. You can, and should, use your code from the previous lab.

The target speedup for Tier 3 is 6x on the full training function with
includes all the above layers.  You'll need to combine multiple
techniques (tiling, threading, renesting) to achieve the target.

## Testing

This lab includes a set of regressions that work just as they did in
the last lab, except that they also cover the new layers used in this
lab.

## Tips

* There are many more things to try in this lab than there have been
  in the earlier labs.  This has two implication:

  * Start early.
 
  * "guess and check" is unlikely to get you a good solution in
    reasonable.  Think carefully about the changes you are making.
    Thinking takes time.  Start early.

  * The autograder servers are going to get very busy near the deadline.  Start early.

* Unfortunately, gprof doesn't work on multi-threaded programs.  You
  can comment out `OPENMP=yes` in `config.env` to make gprof work
  properly.

* OpenMP is a big library.  We've covered a bit of it.  You're free to use the rest.

* There's lot of resources on the web about OpenMP.  Many (or most) of
  them are bad.  This one is pretty good:
  http://jakascorner.com/blog/.  Especially these entries:

  * http://jakascorner.com/blog/2016/04/omp-introduction.html
  
  * http://jakascorner.com/blog/2016/05/omp-for.html
  
  * http://jakascorner.com/blog/2016/07/omp-critical.html

  * http://jakascorner.com/blog/2016/06/omp-for-scheduling.html

  * http://jakascorner.com/blog/2016/06/omp-data-sharing-attributes.html

  * http://jakascorner.com/blog/2016/07/omp-default-none-and-const.html

