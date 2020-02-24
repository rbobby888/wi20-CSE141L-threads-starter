#include "archlab.hpp"
#include <cstdlib>
#include <getopt.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include "CNN/canela.hpp"
#include <opt_cnn.hpp>
extern model_t * build_model(const dataset_t & ds);
void run_component_performance_tests(const dataset_t & ds, int clock_rate, int batch_size, int reps);
void run_canary(int clock_rate);
extern void train_model(model_t * model, dataset_t & train, int epochs, int batch_size=2);
extern double test_model(model_t * model, dataset_t & test, int batch_size=2);
extern double compare_model(model_t * model, model_t * batch_model, dataset_t & test, int batch_size=2);

using namespace std;

#if (0)
int *openmp_example() {
	int * array = new int[1024*1024*1024];

	for (int j = 0; j < 10; j++) {
#pragma omp parallel for 
		for(int i= 0 ; i < 1024*1024*1024; i++) {
			array[i]+= i*j;
		}
	}
	
	return array;
}
#endif

void run_canary(int clock_rate)
{
	pristine_machine();
	set_cpu_clock_frequency(clock_rate);
	theDataCollector->disable_prefetcher();
	ArchLabTimer timer; // create it.
	timer.attr("function", "_canary");
	timer.go();
	archlab_canary(100000000);
}

int main(int argc, char *argv[])
{

	// Parse the command line.

	// They can select datasets and input sizes via the command
	// line in addition to all the performanec counter stuff
	std::vector<int> mhz_s;
	std::vector<int> default_mhz;
	int per_function_reps = 1;
	int train_reps = 1;
	load_frequencies();
	default_mhz.push_back(cpu_frequencies_array[0]);
	std::stringstream clocks;
	for(int i =0; cpu_frequencies_array[i] != 0; i++) {
		clocks << cpu_frequencies_array[i] << " ";
	}
	std::stringstream fastest;
	fastest << cpu_frequencies_array[0];
	archlab_add_option<std::vector<int> >("MHz",
					      mhz_s,
					      default_mhz,
					      fastest.str(),
					      "Which clock rate to run.  Possibilities on this machine are: " + clocks.str());
	archlab_add_option<int>("reps",
				per_function_reps,
				1,
				"1",
				"How many reps of the per-function perf tests to run");
	archlab_add_option<int>("train-reps",
				train_reps,
				1,
				"1",
				"How many reps of the trainiing test to run");

	std::vector<std::string> dataset_s;

	int scale_factor;
	
	std::vector<std::string> default_set;
	default_set.push_back("mnist");

	archlab_add_option<int>("scale", scale_factor, 10, "The scale factor.  Bigger (smaller) numbers mean longer (shorter) run times by running more samples.  The default is 10, which should allow optimized code to run to completion without timing out.  If you want to run without opts, turn it down.");

	archlab_add_option<std::vector<std::string> >("dataset",
						      dataset_s,
						      default_set,
						      "mnist",
						      "Which dataset to use: 'mnist', 'emnist', 'cifar10', 'cifar100', or 'imagenet'. "
						      "Pass it multiple times to run multiple datasets.");
	archlab_parse_cmd_line(&argc, argv);

	
	for(auto & ds: dataset_s) {
		std::cout << "Running " << ds << "\n";

			
		dataset_t *train = new dataset_t;
		dataset_t *test = new dataset_t;
	
		if (ds == "mnist") {
			*train = dataset_t::read(std::string(std::getenv("CANELA_ROOT")) + "/datasets/mnist/mnist-train.dataset", 200 * scale_factor);
			*test = dataset_t::read(std::string(std::getenv("CANELA_ROOT")) + "/datasets/mnist/mnist-test.dataset", 200 * scale_factor);
		} else if (ds == "emnist") {
			*train = dataset_t::read(std::string(std::getenv("CANELA_ROOT")) + "/datasets/mnist/emnist-byclass-train.dataset", 200 * scale_factor);
			*test = dataset_t::read(std::string(std::getenv("CANELA_ROOT")) + "/datasets/mnist/emnist-byclass-test.dataset", 200 * scale_factor);
		} else if (ds == "cifar10") {
			*train = dataset_t::read(std::string(std::getenv("CANELA_ROOT")) + "/datasets/cifar/cifar10_data_batch_1.dataset", 100 * scale_factor);
			*test = dataset_t::read(std::string(std::getenv("CANELA_ROOT")) + "/datasets/cifar/cifar10_test_batch.dataset", 100 * scale_factor);
		} else if (ds == "cifar100") {
			*train = dataset_t::read(std::string(std::getenv("CANELA_ROOT")) + "/datasets/cifar/cifar100_train.dataset", 100 * scale_factor);
			*test = dataset_t::read(std::string(std::getenv("CANELA_ROOT")) + "/datasets/cifar/cifar100_test.dataset", 100 * scale_factor);
		} else if (ds == "imagenet") {
			*train = dataset_t::read(std::string(std::getenv("CANELA_ROOT")) + "/datasets/imagenet/imagenet.dataset", 1 * scale_factor);
			*test = dataset_t::read(std::string(std::getenv("CANELA_ROOT")) + "/datasets/imagenet/imagenet.dataset", 1 * scale_factor);
		} else if (ds == "mininet") {
			*train = dataset_t::read(std::string(std::getenv("CANELA_ROOT")) + "/datasets/imagenet/mininet.dataset", 1 * scale_factor);
			*test = dataset_t::read(std::string(std::getenv("CANELA_ROOT")) + "/datasets/imagenet/mininet.dataset", 1 * scale_factor);
		}

		theDataCollector->register_tag("dataset", ds);
		theDataCollector->register_tag("training_inputs_count", train->test_cases.size());
		theDataCollector->register_tag("test_inputs_count", test->test_cases.size());


		run_canary(mhz_s[0]);
		run_component_performance_tests(*test, mhz_s[0], 16, per_function_reps);

		
		model_t * model = build_model(*train);
		int batch_size = 4;
		model->change_batch_size(batch_size);


		std::cout << model->geometry() << "\n"; // output a summary of its sturcture and size.
		std::cout << "Training data size: " << (train->get_total_memory_size()+0.0)/(1024*1024)  << " MB" << std::endl;
		std::cout << "Training cases    : " << train->size() << std::endl;
		std::cout << "Testing data size : " << (test->get_total_memory_size()+0.0)/(1024*1024)  << " MB" << std::endl;
		std::cout << "Testing cases     : " << test->size() << std::endl;

		std::cout << "Regression parameters\n" <<model->regression_code() << std::endl;
		
	// 	{
	// 		ArchLabTimer timer;
	// 		pristine_machine();
	// 		set_cpu_clock_frequency(mhz_s[0]);
	// 		theDataCollector->disable_prefetcher();
	// 		timer.attr("function", "openmp_example");
	// 		timer.go();
	// 		int * t = openmp_example();
	// 		delete t;
	// 	}


#if (1)
		// Timing occurs inside here
		train_model(model, *train, train_reps, batch_size);
#endif
		
#if (0)
		double total_error;
		{
			ArchLabTimer timer; // create it.
			pristine_machine();
			set_cpu_clock_frequency(mhz_s[0]);
			theDataCollector->disable_prefetcher();
			timer.go();
			train_model_full(model, *train, 2, batch_size);
			total_error = test_model(model, *train, batch_size);
		}
		std::cout << "Classification accuracy: " << total_error << "\n";
#endif
		delete test;
		delete train;

	}
	
	archlab_write_stats();
	return 0;
}

model_t * build_model(const dataset_t & ds)  {
	model_t * model = new model_t();
	layer_t *layer1 = new opt_conv_layer_t( 1, 20, 5, 1, ds.data_size );
	layer_t *layer2 = new opt_pool_layer_t( 3, 5, 1, layer1->out.size );
	layer_t * layer3 = new opt_relu_layer_t( layer2->out.size );
	layer_t *layer4 = new opt_fc_layer_t( layer3->out.size, ds.label_size.x );
	// Layers build up sequentially.
	// You have to start with the
	// input layer and finish with the
	// output.
	model->add_layer( *layer1 );
	model->add_layer( *layer2 );
	model->add_layer( *layer3 );
	model->add_layer( *layer4 );
	return model;
}


void train_model(model_t * model,
		 dataset_t & train,
		 int reps,
		 int batch_size) {
	int batch_index = 0;
	tensor_t<double> batch_data(tdsize(train.data_size.x, train.data_size.y, train.data_size.z, batch_size));
	tensor_t<double> batch_label(tdsize(train.label_size.x, train.label_size.y, train.label_size.z, batch_size));
	for (auto& t : train.test_cases ) {
		for (int x = 0; x < t.data.size.x; x += 1)
			for (int y = 0; y < t.data.size.y; y += 1)
				for (int z = 0; z < t.data.size.z; z += 1)
					batch_data(x, y, z, batch_index) = t.data(x, y, z); 		
		for (int x = 0; x < t.label.size.x; x += 1)
			for (int y = 0; y < t.label.size.y; y += 1)
				for (int z = 0; z < t.label.size.z; z += 1)
					batch_label(x, y, z, batch_index) = t.label(x, y, z); 		
		batch_index += 1;
		if (batch_index >= batch_size) {
			batch_index = 0;
			{
				ArchLabTimer timer; // create it.
				pristine_machine();
				theDataCollector->disable_prefetcher();
				timer.attr("function", "train_model");
				timer.go();
				for (int i = 0; i < reps; i += 1) {
					model->train(batch_data, batch_label);
				}
				return;
			}
		}
	}
	
}

void train_model_full(model_t * model,
		 dataset_t & train,
		 int epochs,
		 int batch_size) {
	int batch_index = 0;
	tensor_t<double> batch_data(tdsize(train.data_size.x, train.data_size.y, train.data_size.z, batch_size));
	tensor_t<double> batch_label(tdsize(train.label_size.x, train.label_size.y, train.label_size.z, batch_size));
	for (int i = 0; i < epochs; i += 1) {
		for (auto& t : train.test_cases ) {
			for (int x = 0; x < t.data.size.x; x += 1)
				for (int y = 0; y < t.data.size.y; y += 1)
					for (int z = 0; z < t.data.size.z; z += 1)
						batch_data(x, y, z, batch_index) = t.data(x, y, z); 		
			for (int x = 0; x < t.label.size.x; x += 1)
				for (int y = 0; y < t.label.size.y; y += 1)
					for (int z = 0; z < t.label.size.z; z += 1)
						batch_label(x, y, z, batch_index) = t.label(x, y, z); 		
			batch_index += 1;
			if (batch_index >= batch_size) {
				batch_index = 0;
				model->train(batch_data, batch_label);
			}
		}
	}

}

double test_model(model_t * model,
	dataset_t & test,
	int batch_size) {
	int correct = 0;
	int incorrect = 0;
	int batch_index = 0;
	std::cout << "Timing model\n";

	tensor_t<double> batch_data(tdsize(test.data_size.x, test.data_size.y, test.data_size.z, batch_size));
	tensor_t<double> batch_label(tdsize(test.label_size.x, test.label_size.y, test.label_size.z, batch_size));
	for (auto& t : test.test_cases ) {
		for (int x = 0; x < t.data.size.x; x += 1)
			for (int y = 0; y < t.data.size.y; y += 1)
				for (int z = 0; z < t.data.size.z; z += 1)
					batch_data(x, y, z, batch_index) = t.data(x, y, z); 		
		for (int x = 0; x < t.label.size.x; x += 1)
			for (int y = 0; y < t.label.size.y; y += 1)
				for (int z = 0; z < t.label.size.z; z += 1)
					batch_label(x, y, z, batch_index) = t.label(x, y, z); 		
		batch_index += 1;
		if (batch_index >= batch_size) {
			batch_index = 0;
			tensor_t<double>& out = model->apply(batch_data);
			std::vector<tdsize> maxes = out.argmax_b();
			std::vector<tdsize> correct_maxes = batch_label.argmax_b();
			for (int i = 0; i < batch_size; i += 1) {
				if (maxes[i].x == correct_maxes[i].x) {
					correct += 1;
				} else {
					incorrect += 1;
				}
			}

		}
	}
	double total_error = (correct+0.0)/(correct+ incorrect +0.0);
	return total_error;
}

void time_activate(layer_t * l, int reps, const std::string& name) {
	ArchLabTimer timer; // create it.
	timer.attr("function", name);
	std::cout << "Timing " << name << "\n";
	tensor_t<double> _in(l->in.size);
	timer.go();
	for (int i = 0; i < reps; i++)
		l->activate(_in);
}

void time_calc_grads(layer_t * l, int reps, const std::string& name) {

	ArchLabTimer timer; // create it.
	timer.attr("function", name);
	std::cout << "Timing " << name << "\n";

	tensor_t<double> _out(l->out.size);
	timer.go();
	for (int i = 0; i < reps; i++)
		l->calc_grads(_out);
}

void time_fix_weights(layer_t * l, int reps, const std::string& name) {

	ArchLabTimer timer; // create it.
	timer.attr("function", name);
	std::cout << "Timing " << name << "\n";
	timer.go();
	for (int i = 0; i < reps; i++)
		l->fix_weights();
}

void run_component_performance_tests(const dataset_t & ds, int clock_rate, int batch_size, int reps)
{

	tdsize in_size = ds.data_size;
	in_size.b = batch_size;
	opt_conv_layer_t c( 1, 20, 5, 1, in_size);

 	opt_pool_layer_t p( 3, 5, 1, c.out.size);

	opt_relu_layer_t r( p.out.size);

	opt_fc_layer_t l( r.out.size, ds.label_size.x );
	
//fc layer
	pristine_machine();
	set_cpu_clock_frequency(clock_rate);
	theDataCollector->disable_prefetcher();
	time_activate(&l, reps, "fc_activate");
	
	pristine_machine();
	set_cpu_clock_frequency(clock_rate);
	theDataCollector->disable_prefetcher();
	time_calc_grads(&l,reps*5, "fc_calc_grads");

	pristine_machine();
	set_cpu_clock_frequency(clock_rate);
	theDataCollector->disable_prefetcher();
	time_fix_weights(&l, reps*4, "fc_fix_weights");

//Conv Layer
	pristine_machine();
	set_cpu_clock_frequency(clock_rate);
	theDataCollector->disable_prefetcher();
	time_activate(&c, (reps/10), "conv_activate");

	pristine_machine();
	set_cpu_clock_frequency(clock_rate);
	theDataCollector->disable_prefetcher();
	time_calc_grads(&c,(reps/15), "conv_calc_grads");

	pristine_machine();
	set_cpu_clock_frequency(clock_rate);
	theDataCollector->disable_prefetcher();
	time_fix_weights(&c, (reps*200), "conv_fix_weights");

//Pool Layer
	pristine_machine();
	set_cpu_clock_frequency(clock_rate);
	theDataCollector->disable_prefetcher();
	time_activate(&p, (reps*40), "pool_activate");

	pristine_machine();
	set_cpu_clock_frequency(clock_rate);
	theDataCollector->disable_prefetcher();
	time_calc_grads(&p, (reps*20), "pool_calc_grads");

//Relu Layer
	pristine_machine();
	set_cpu_clock_frequency(clock_rate);
	theDataCollector->disable_prefetcher();
	time_activate(&r, (reps*400), "relu_activate");

	pristine_machine();
	set_cpu_clock_frequency(clock_rate);
	theDataCollector->disable_prefetcher();
	time_calc_grads(&r,(reps*200), "relu_calc_grads");
}
