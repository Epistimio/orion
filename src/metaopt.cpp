#include <boost/program_options.hpp>

// Global headers
//
#include <iostream>
#include <string>

// Local headers

namespace
{
  const size_t ERROR_IN_COMMAND_LINE = 1; 
  const size_t SUCCESS = 0; 
  const size_t ERROR_UNHANDLED_EXCEPTION = 2; 

} // namespace 

int main(int argc, char** argv) 
{
  try
  {
    /** Define and parse the program options 
     */
    namespace po = boost::program_options;
    po::options_description desc("Options");
    desc.add_options()
      ("--script", "Python Script file (Experiment)"),
      ("--hyperparameter-optimization-argparse", "Normal Python Script Argparse - Hyperparameters, e.g. --batchsize --dropout"); 
 
    po::variables_map vm;
    try 
    { 
      po::store(po::parse_command_line(argc, argv, desc),  
                vm); // can throw 
 
      /** --help option 
       */ 
      if ( vm.count("help")  ) 
      { 
        std::cout << "Meta Hyperparameter Optimization" << std::endl  << desc << std::endl; 
        return SUCCESS; 
      } 
 
      po::notify(vm); // throws on error, so do after help in case 
                      // there are any problems 
    } 
    catch(po::error& e) 
    { 
      std::cerr << "ERROR: " << e.what() << std::endl << std::endl; 
      std::cerr << desc << std::endl; 
      return ERROR_IN_COMMAND_LINE; 
    } 
 
    // application code here // 
 
  } 
  catch(std::exception& e) 
  { 
    std::cerr << "Unhandled Exception reached the top of main: " 
              << e.what() << ", application will now exit" << std::endl; 
    return ERROR_UNHANDLED_EXCEPTION; 
 
  } 
 
  return SUCCESS; 
 
} // main 
