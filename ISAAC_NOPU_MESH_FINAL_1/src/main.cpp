// $Id$

/*
 Copyright (c) 2007-2015, Trustees of The Leland Stanford Junior University
 All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:

 Redistributions of source code must retain the above copyright notice, this 
 list of conditions and the following disclaimer.
 Redistributions in binary form must reproduce the above copyright notice, this
 list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE 
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
 ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/*main.cpp
 *
 *The starting point of the network simulator
 *-Include all network header files
 *-initilize the network
 *-initialize the traffic manager and set it to run
 *
 *
 */
#include <sys/time.h>
#include <sys/wait.h>
#include <unistd.h>
#include <string>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include "booksim.hpp"
#include "routefunc.hpp"
#include "traffic.hpp"
#include "booksim_config.hpp"
#include "trafficmanager.hpp"
#include "random_utils.hpp"
#include "network.hpp"
#include "injection.hpp"
#include "power_module.hpp"

using namespace std;

///////////////////////////////////////////////////////////////////////////////
//Global declarations
//////////////////////

 /* the current traffic manager instance */
TrafficManager * trafficManager = NULL;

int GetSimTime() {
  return trafficManager->getTime();
}

class Stats;
Stats * GetStats(const std::string & name) {
  Stats* test =  trafficManager->getStats(name);
  if(test == 0){
    cout<<"warning statistics "<<name<<" not found"<<endl;
  }
  return test;
}

/* printing activity factor*/
bool gPrintActivity;

int gK;//radix
int gN;//dimension
int gC;//concentration

int gNodes;
int input_num;
//generate nocviewer trace
bool gTrace;

ostream * gWatchOut;
// ofstream outFile;
// ofstream outFile_leaveflit;

/////////////////////////////////////////////////////////////////////////////

bool Simulate( BookSimConfig const & config )
{
  vector<Network *> net;

  int subnets = config.GetInt("subnets");
  /*To include a new network, must register the network here
   *add an else if statement with the name of the network
   */
  net.resize(subnets);
  for (int i = 0; i < subnets; ++i) {
    ostringstream name;
    name << "network_" << i;
    net[i] = Network::New( config, name.str() );
  }

  /*tcc and characterize are legacy
   *not sure how to use them 
   */

  assert(trafficManager == NULL);
  trafficManager = TrafficManager::New( config, net ) ;

  /*Start the simulation run
   */

  double total_time; /* Amount of time we've run */
  struct timeval start_time, end_time; /* Time before/after user code */
  total_time = 0.0;
  gettimeofday(&start_time, NULL);

  bool result = trafficManager->Run() ;


  gettimeofday(&end_time, NULL);
  total_time = ((double)(end_time.tv_sec) + (double)(end_time.tv_usec)/1000000.0)
            - ((double)(start_time.tv_sec) + (double)(start_time.tv_usec)/1000000.0);

  cout<<"Total run time "<<total_time<<endl;

  for (int i=0; i<subnets; ++i) {

    ///Power analysis
    if(config.GetInt("sim_power") > 0){
      Power_Module pnet(net[i], config);
      pnet.run();
    }

    delete net[i];
  }

  delete trafficManager;
  trafficManager = NULL;

  return result;
}

void RunSimulation(const char *config_file, int instance_id) {
  cout << "RunSimulation: Start " << instance_id << " with config file: " << config_file << endl;
  BookSimConfig config;
  input_num = 0;

  const char *default_config_file = config_file;
  int argc = 2;
  char **argv = new char *[argc];
  argv[0] = const_cast<char *>("simulation");
  argv[1] = new char[strlen(default_config_file) + 1];
  strcpy(argv[1], default_config_file);

  if (!ParseArgs(&config, argc, argv)) {
    cerr << "Usage: " << argv[0] << " configfile... [param=value...]" << endl;
    return;
  }

  InitializeRoutingMap(config);

  gPrintActivity = (config.GetInt("print_activity") > 0);
  gTrace = (config.GetInt("viewer_trace") > 0);

  string watch_out_file = "./ISAAC_NOPU_MESH_FINAL_1/src/output_file/watch_file_" + to_string(instance_id);
  if (watch_out_file == "") {
    gWatchOut = NULL;
  } else if (watch_out_file == "-") {
    gWatchOut = &cout;
  } else {
    gWatchOut = new ofstream(watch_out_file.c_str());
  }

  bool result = Simulate(config);

  delete[] argv[1];
  delete[] argv;

  cout << "RunSimulation: End " << instance_id << endl;
}

int main(int argc, char **argv) {
  const int num_processes = 5;
  pid_t pids[num_processes];

  for (int i = 0; i < num_processes; ++i) {
    if ((pids[i] = fork()) < 0) {
      cerr << "Fork failed for process " << i << endl;
      return 1;
    } else if (pids[i] == 0) {
      // Child process
      string config_file = "./ISAAC_NOPU_MESH_FINAL_1/src/examples/mesh88_lat_" + to_string(i);
      RunSimulation(config_file.c_str(), i);
      exit(0); // Ensure child process exits after running simulation
    }
  }

  // Parent process waits for all child processes to finish
  for (int i = 0; i < num_processes; ++i) {
    waitpid(pids[i], NULL, 0);
  }

  return 0;
}
