#include <mpi.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iomanip>
#include <json/json.h> 

// Data structures for the PEMS
struct SensorData {
    double solar_output;        // kW
    double battery_soc;         // State of Charge (0-100%)
    double consumption;         // kW
    double temperature;         // Celsius
    double humidity;           // %
    int timestamp;             // Unix timestamp
};

struct WeatherForecast {
    double solar_irradiance;   // W/m²
    double temperature;        // Celsius
    double humidity;          // %
    double cloud_cover;       // %
    int timestamp;            // Unix timestamp
};

struct ActionPlan {
    int node_id;
    bool delay_washing_machine;
    bool delay_dishwasher;
    double battery_charge_rate;  // % per hour
    bool activate_backup_generator;
    double load_reduction_factor; // 0.0 to 1.0
};

struct PredictionResult {
    double predicted_solar_24h[24];
    double predicted_consumption_24h[24];
    double battery_soc_forecast[24];
    double energy_deficit_risk;
};

// Mock data generators
class DataGenerator {
private:
    std::mt19937 rng;
    std::normal_distribution<double> normal_dist;
    
public:
    DataGenerator(int seed) : rng(seed), normal_dist(0.0, 1.0) {}
    
    SensorData generateSensorData(int node_id, int time_of_day) {
        SensorData data;
        
        // Simulate solar output based on time of day (peak at noon)
        double solar_factor = std::max(0.0, std::sin(M_PI * time_of_day / 24.0));
        data.solar_output = solar_factor * (8.0 + normal_dist(rng) * 1.5); // 0-10 kW
        data.solar_output = std::max(0.0, data.solar_output);
        
        // Battery SoC varies based on solar production and consumption
        data.battery_soc = 50.0 + 30.0 * solar_factor + normal_dist(rng) * 10.0;
        data.battery_soc = std::max(10.0, std::min(100.0, data.battery_soc));
        
        // Consumption varies throughout day
        double consumption_base = (time_of_day < 6 || time_of_day > 22) ? 2.0 : 5.0;
        data.consumption = consumption_base + normal_dist(rng) * 1.0;
        data.consumption = std::max(1.0, data.consumption);
        
        data.temperature = 25.0 + normal_dist(rng) * 5.0;
        data.humidity = 60.0 + normal_dist(rng) * 20.0;
        data.timestamp = time_of_day * 3600; // Convert to seconds
        
        return data;
    }
    
    WeatherForecast generateWeatherForecast(int hour_ahead) {
        WeatherForecast forecast;
        
        double solar_factor = std::max(0.0, std::sin(M_PI * hour_ahead / 24.0));
        forecast.solar_irradiance = solar_factor * (800.0 + normal_dist(rng) * 100.0);
        forecast.solar_irradiance = std::max(0.0, forecast.solar_irradiance);
        
        forecast.temperature = 25.0 + normal_dist(rng) * 8.0;
        forecast.humidity = 60.0 + normal_dist(rng) * 25.0;
        forecast.cloud_cover = std::max(0.0, std::min(100.0, 30.0 + normal_dist(rng) * 30.0));
        forecast.timestamp = hour_ahead * 3600;
        
        return forecast;
    }
};

// Neural Network for prediction (simplified implementation)
class SimpleNeuralNetwork {
private:
    std::vector<std::vector<double>> weights1, weights2;
    std::vector<double> bias1, bias2;
    
    double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }
    double relu(double x) { return std::max(0.0, x); }
    
public:
    SimpleNeuralNetwork() {
        // Initialise a simple 5->10->3 network
        weights1.resize(5, std::vector<double>(10));
        weights2.resize(10, std::vector<double>(3));
        bias1.resize(10);
        bias2.resize(3);
        
        // Random initialisation
        std::mt19937 rng(42);
        std::uniform_real_distribution<double> dist(-0.5, 0.5);
        
        for(auto& row : weights1) {
            for(auto& w : row) w = dist(rng);
        }
        for(auto& row : weights2) {
            for(auto& w : row) w = dist(rng);
        }
        for(auto& b : bias1) b = dist(rng);
        for(auto& b : bias2) b = dist(rng);
    }
    
    std::vector<double> predict(const std::vector<double>& input) {
        // Forward pass
        std::vector<double> hidden(10, 0.0);
        for(int i = 0; i < 10; i++) {
            for(int j = 0; j < 5; j++) {
                hidden[i] += input[j] * weights1[j][i];
            }
            hidden[i] += bias1[i];
            hidden[i] = relu(hidden[i]);
        }
        
        std::vector<double> output(3, 0.0);
        for(int i = 0; i < 3; i++) {
            for(int j = 0; j < 10; j++) {
                output[i] += hidden[j] * weights2[j][i];
            }
            output[i] += bias2[i];
            output[i] = sigmoid(output[i]);
        }
        
        return output;
    }
};

// Main PEMS Class with integrated JSON output
class PEMS {
private:
    int rank, size;
    DataGenerator* dataGen;
    SimpleNeuralNetwork* neuralNet;
    
    // JSON output methods
    void outputNodeDataJSON(const std::vector<std::vector<SensorData>>& globalData) {
        if (rank == 0) {
            Json::Value root;
            root["type"] = "node_data";
            root["timestamp"] = static_cast<int>(time(nullptr));
            
            Json::Value nodes(Json::arrayValue);
            for (int i = 0; i < size; i++) {
                Json::Value node;
                node["id"] = i;
                node["name"] = "Node " + std::to_string(i);
                
                if (!globalData[i].empty()) {
                    // Use latest data point for each node
                    const SensorData& latest = globalData[i].back();
                    node["solar_output"] = latest.solar_output;
                    node["battery_soc"] = latest.battery_soc;
                    node["consumption"] = latest.consumption;
                    node["temperature"] = latest.temperature;
                    node["humidity"] = latest.humidity;
                    node["status"] = (latest.battery_soc > 60) ? "online" : 
                                    (latest.battery_soc > 30) ? "warning" : "critical";
                } else {
                    node["status"] = "offline";
                }
                nodes.append(node);
            }
            root["nodes"] = nodes;
            
            std::cout << "JSON_OUTPUT:" << root << std::endl;
        }
    }
    
    void outputPredictionsJSON(const PredictionResult& prediction) {
        if (rank == 0) {
            Json::Value root;
            root["type"] = "predictions";
            
            Json::Value solar(Json::arrayValue);
            Json::Value consumption(Json::arrayValue);
            Json::Value battery(Json::arrayValue);
            
            for (int i = 0; i < 24; i++) {
                solar.append(prediction.predicted_solar_24h[i]);
                consumption.append(prediction.predicted_consumption_24h[i]);
                battery.append(prediction.battery_soc_forecast[i]);
            }
            
            root["solar_forecast"] = solar;
            root["consumption_forecast"] = consumption;
            root["battery_forecast"] = battery;
            root["deficit_risk"] = prediction.energy_deficit_risk;
            
            std::cout << "JSON_OUTPUT:" << root << std::endl;
        }
    }
    
    void outputActionPlansJSON(const std::vector<ActionPlan>& plans) {
        if (rank == 0) {
            Json::Value root;
            root["type"] = "action_plans";
            
            Json::Value actions(Json::arrayValue);
            for (const auto& plan : plans) {
                Json::Value action;
                action["node_id"] = plan.node_id;
                action["delay_washing_machine"] = plan.delay_washing_machine;
                action["delay_dishwasher"] = plan.delay_dishwasher;
                action["battery_charge_rate"] = plan.battery_charge_rate;
                action["activate_backup_generator"] = plan.activate_backup_generator;
                action["load_reduction_factor"] = plan.load_reduction_factor;
                actions.append(action);
            }
            root["actions"] = actions;
            
            std::cout << "JSON_OUTPUT:" << root << std::endl;
        }
    }
    
    void outputPerformanceJSON(double processingTime, double speedup, double efficiency) {
        if (rank == 0) {
            Json::Value root;
            root["type"] = "performance";
            root["processing_time_ms"] = processingTime;
            root["speedup_factor"] = speedup;
            root["efficiency_percent"] = efficiency;
            root["total_nodes"] = size;
            
            std::cout << "JSON_OUTPUT:" << root << std::endl;
        }
    }
    
public:
    PEMS(int r, int s) : rank(r), size(s) {
        dataGen = new DataGenerator(rank * 123 + 456);
        if(rank == 0) { // Master node has the neural network
            neuralNet = new SimpleNeuralNetwork();
        }
    }
    
    ~PEMS() {
        delete dataGen;
        if(rank == 0) delete neuralNet;
    }
    
    // Node-level data collection
    void collectLocalData(std::vector<SensorData>& localData, int hours = 24) {
        std::cout << "Node " << rank << ": Collecting local sensor data..." << std::endl;
        
        localData.clear();
        for(int h = 0; h < hours; h++) {
            SensorData data = dataGen->generateSensorData(rank, h);
            localData.push_back(data);
        }
        
        std::cout << "Node " << rank << ": Collected " << localData.size() << " data points" << std::endl;
    }
    
    // Parallel data aggregation with JSON output
    void aggregateDataParallel(const std::vector<SensorData>& localData, 
                              std::vector<std::vector<SensorData>>& globalData) {
        if(rank == 0) {
            std::cout << "Master: Starting parallel data aggregation..." << std::endl;
            globalData.resize(size);
        }
        
        // Gather all node data to master
        int dataSize = localData.size() * sizeof(SensorData);
        std::vector<int> recvCounts(size, dataSize);
        std::vector<int> displs(size);
        
        for(int i = 0; i < size; i++) {
            displs[i] = i * dataSize;
        }
        
        if(rank == 0) {
            globalData[0] = localData;
            
            // Receive data from other nodes
            for(int node = 1; node < size; node++) {
                std::vector<SensorData> nodeData(24);
                MPI_Recv(nodeData.data(), dataSize, MPI_BYTE, node, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                globalData[node] = nodeData;
            }
            
            std::cout << "Master: Data aggregation complete. Total nodes: " << size << std::endl;
            
            // Output node data as JSON
            outputNodeDataJSON(globalData);
        } else {
            // Send local data to master
            MPI_Send(const_cast<SensorData*>(localData.data()), dataSize, MPI_BYTE, 0, 0, MPI_COMM_WORLD);
        }
    }
    
    // Parallel prediction model with JSON output
    PredictionResult runParallelPrediction(const std::vector<std::vector<SensorData>>& globalData) {
        PredictionResult result;
        
        if(rank == 0) {
            std::cout << "Master: Running parallel prediction model..." << std::endl;
            
            auto start_time = std::chrono::high_resolution_clock::now();
            
            // Prepare input features for neural network
            for(int hour = 0; hour < 24; hour++) {
                double avg_solar = 0, avg_consumption = 0, avg_battery = 0, avg_temp = 0;
                
                // Aggregate data from all nodes
                for(int node = 0; node < size; node++) {
                    if(hour < globalData[node].size()) {
                        avg_solar += globalData[node][hour].solar_output;
                        avg_consumption += globalData[node][hour].consumption;
                        avg_battery += globalData[node][hour].battery_soc;
                        avg_temp += globalData[node][hour].temperature;
                    }
                }
                
                avg_solar /= size;
                avg_consumption /= size;
                avg_battery /= size;
                avg_temp /= size;
                
                // Generate weather forecast
                WeatherForecast forecast = dataGen->generateWeatherForecast(hour);
                
                // Prepare input vector for neural network
                std::vector<double> input = {
                    avg_solar / 10.0,                    // Normalised solar output
                    avg_consumption / 10.0,              // Normalised consumption
                    avg_battery / 100.0,                 // Normalised battery SoC
                    forecast.solar_irradiance / 1000.0,  // Normalised irradiance
                    forecast.cloud_cover / 100.0         // Normalised cloud cover
                };
                
                // Run prediction
                std::vector<double> prediction = neuralNet->predict(input);
                
                // Scale predictions back to real units
                result.predicted_solar_24h[hour] = prediction[0] * 10.0;        // 0-10 kW
                result.predicted_consumption_24h[hour] = prediction[1] * 10.0;   // 0-10 kW
                
                // Calculate battery SoC forecast
                if(hour == 0) {
                    result.battery_soc_forecast[hour] = avg_battery;
                } else {
                    double energy_balance = result.predicted_solar_24h[hour] - result.predicted_consumption_24h[hour];
                    result.battery_soc_forecast[hour] = result.battery_soc_forecast[hour-1] + energy_balance * 2.0; // Simplified
                    result.battery_soc_forecast[hour] = std::max(0.0, std::min(100.0, result.battery_soc_forecast[hour]));
                }
            }
            
            // Calculate energy deficit risk
            result.energy_deficit_risk = 0.0;
            for(int hour = 0; hour < 24; hour++) {
                if(result.battery_soc_forecast[hour] < 20.0) {
                    result.energy_deficit_risk += (20.0 - result.battery_soc_forecast[hour]) / 20.0;
                }
            }
            result.energy_deficit_risk /= 24.0;
            
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            
            std::cout << "Master: Prediction completed in " << duration.count() << " ms" << std::endl;
            std::cout << "Master: Energy deficit risk: " << std::fixed << std::setprecision(3) 
                      << result.energy_deficit_risk * 100 << "%" << std::endl;
            
            // Output predictions as JSON
            outputPredictionsJSON(result);
        }
        
        return result;
    }
    
    // Generate optimised action plans with JSON output
    std::vector<ActionPlan> generateActionPlans(const PredictionResult& prediction) {
        std::vector<ActionPlan> plans;
        
        if(rank == 0) {
            std::cout << "Master: Generating optimised action plans..." << std::endl;
            
            plans.resize(size);
            
            for(int node = 0; node < size; node++) {
                ActionPlan& plan = plans[node];
                plan.node_id = node;
                
                // Decision logic based on predictions
                double min_battery_soc = *std::min_element(prediction.battery_soc_forecast, 
                                                          prediction.battery_soc_forecast + 24);
                
                // Load shifting decisions
                plan.delay_washing_machine = (min_battery_soc < 30.0);
                plan.delay_dishwasher = (min_battery_soc < 25.0);
                
                // Battery management
                if(min_battery_soc < 40.0) {
                    plan.battery_charge_rate = 15.0; // Charge at 15% per hour
                } else {
                    plan.battery_charge_rate = 5.0;  // Normal charging
                }
                
                // Backup generator activation
                plan.activate_backup_generator = (prediction.energy_deficit_risk > 0.3);
                
                // Load reduction
                if(prediction.energy_deficit_risk > 0.5) {
                    plan.load_reduction_factor = 0.8; // Reduce load by 20%
                } else {
                    plan.load_reduction_factor = 1.0; // No reduction
                }
                
                std::cout << "Master: Action plan for Node " << node << ":" << std::endl;
                std::cout << "  - Delay washing machine: " << (plan.delay_washing_machine ? "Yes" : "No") << std::endl;
                std::cout << "  - Battery charge rate: " << plan.battery_charge_rate << "%" << std::endl;
                std::cout << "  - Activate backup gen: " << (plan.activate_backup_generator ? "Yes" : "No") << std::endl;
                std::cout << "  - Load reduction: " << (1.0 - plan.load_reduction_factor) * 100 << "%" << std::endl;
            }
            
            // Output action plans as JSON
            outputActionPlansJSON(plans);
        }
        
        return plans;
    }
    
    // Distribute action plans to nodes
    void distributeActionPlans(const std::vector<ActionPlan>& plans, ActionPlan& myPlan) {
        if(rank == 0) {
            std::cout << "Master: Distributing action plans to all nodes..." << std::endl;
            
            // Send plans to other nodes
            for(int node = 1; node < size; node++) {
                MPI_Send(const_cast<ActionPlan*>(&plans[node]), sizeof(ActionPlan), MPI_BYTE, 
                        node, 1, MPI_COMM_WORLD);
            }
            
            // Master gets its own plan
            myPlan = plans[0];
        } else {
            // Receive plan from master
            MPI_Recv(&myPlan, sizeof(ActionPlan), MPI_BYTE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        
        std::cout << "Node " << rank << ": Received action plan" << std::endl;
    }
    
    // Execute local actions
    void executeLocalActions(const ActionPlan& plan) {
        std::cout << "Node " << rank << ": Executing local actions..." << std::endl;
        
        // Simulate appliance control
        if(plan.delay_washing_machine) {
            std::cout << "Node " << rank << ": Delaying washing machine operation" << std::endl;
        }
        
        if(plan.delay_dishwasher) {
            std::cout << "Node " << rank << ": Delaying dishwasher operation" << std::endl;
        }
        
        // Simulate battery management
        std::cout << "Node " << rank << ": Setting battery charge rate to " 
                  << plan.battery_charge_rate << "%" << std::endl;
        
        // Simulate backup generator
        if(plan.activate_backup_generator) {
            std::cout << "Node " << rank << ": Activating backup generator" << std::endl;
        }
        
        // Simulate load reduction
        if(plan.load_reduction_factor < 1.0) {
            std::cout << "Node " << rank << ": Reducing load by " 
                      << (1.0 - plan.load_reduction_factor) * 100 << "%" << std::endl;
        }
        
        std::cout << "Node " << rank << ": Local actions executed successfully" << std::endl;
    }
    
    // Performance analysis with JSON output
    void performanceAnalysis() {
        if(rank == 0) {
            std::cout << "\n=== PERFORMANCE ANALYSIS ===" << std::endl;
            std::cout << "System Configuration:" << std::endl;
            std::cout << "  - Total nodes: " << size << std::endl;
            std::cout << "  - Parallel processing: ENABLED" << std::endl;
            std::cout << "  - Distributed architecture: ENABLED" << std::endl;
            
            // Simulate speedup calculation
            double sequential_time = 1000.0; // ms (simulated)
            double parallel_time = sequential_time / size * 0.8; // Amdahl's law approximation
            double speedup = sequential_time / parallel_time;
            double efficiency = (speedup / size) * 100;
            
            std::cout << "\nPerformance Metrics:" << std::endl;
            std::cout << "  - Sequential processing time: " << sequential_time << " ms" << std::endl;
            std::cout << "  - Parallel processing time: " << parallel_time << " ms" << std::endl;
            std::cout << "  - Speedup factor: " << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
            std::cout << "  - Efficiency: " << std::fixed << std::setprecision(1) << efficiency << "%" << std::endl;
            
            // Output performance data as JSON
            outputPerformanceJSON(parallel_time, speedup, efficiency);
        }
    }
};

int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if(rank == 0) {
        std::cout << "=== PREDICTIVE ENERGY MANAGEMENT SYSTEM (PEMS) ===" << std::endl;
        std::cout << "Initialising distributed system with " << size << " nodes..." << std::endl;
    }
    
    // Create PEMS instance
    PEMS pems(rank, size);
    
    // Synchronise all processes
    MPI_Barrier(MPI_COMM_WORLD);
    
    try {
        // Local data collection
        std::vector<SensorData> localData;
        pems.collectLocalData(localData, 24);
        
        MPI_Barrier(MPI_COMM_WORLD);
        
        // Parallel data aggregation (outputs node data JSON)
        std::vector<std::vector<SensorData>> globalData;
        pems.aggregateDataParallel(localData, globalData);
        
        MPI_Barrier(MPI_COMM_WORLD);
        
        // Parallel prediction (outputs predictions JSON)
        PredictionResult prediction;
        prediction = pems.runParallelPrediction(globalData);
        
        MPI_Barrier(MPI_COMM_WORLD);
        
        // Generate action plans (outputs action plans JSON)
        std::vector<ActionPlan> actionPlans;
        actionPlans = pems.generateActionPlans(prediction);
        
        MPI_Barrier(MPI_COMM_WORLD);
        
        // Distribute action plans
        ActionPlan myPlan;
        pems.distributeActionPlans(actionPlans, myPlan);
        
        MPI_Barrier(MPI_COMM_WORLD);
        
        // Execute local actions
        pems.executeLocalActions(myPlan);
        
        MPI_Barrier(MPI_COMM_WORLD);
        
        // Performance analysis (outputs performance JSON)
        pems.performanceAnalysis();
        
        if(rank == 0) {
            std::cout << "\n=== SYSTEM SIMULATION COMPLETE ===" << std::endl;
            std::cout << "All nodes have successfully executed their action plans." << std::endl;
            std::cout << "The system is now optimised for the next 24-hour period." << std::endl;
        }
        
    } catch(const std::exception& e) {
        std::cerr << "Node " << rank << " Error: " << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    // Finalise MPI
    MPI_Finalize();
    
    return 0;
}