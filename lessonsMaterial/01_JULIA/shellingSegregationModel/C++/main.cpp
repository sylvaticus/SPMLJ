#include <iostream>
#include <unistd.h>
#include <time.h>
#include <cstdlib>

#include <string>
#include <vector>

#include "agent.h"
#include "env.h"

using namespace std;




int main() {
    cout << "!! Benvenuto nel programma di simulazione del Modello di segregazione di Schelling !!" << endl << endl;
    srand(time(NULL));
    env* ENV = new env;
    ENV->init();

    for (int i=0;i< ENV->getSteps(); i++){
        ENV->print(i);
        ENV->step(i);
        sleep(ENV->getWaitSeconds()); // seconds

    }
    delete ENV;
    return 0;
}



