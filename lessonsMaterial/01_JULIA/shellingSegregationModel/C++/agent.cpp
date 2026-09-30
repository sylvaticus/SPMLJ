#include <algorithm>
#include "agent.h"

agent::agent(){
    //ctor
}

agent::~agent(){
    //dtor
}


bool
agent::isHappy(){
 return this->isHappy(homeX, homeY);
}

bool
agent::isHappy(int x, int y){
    int neighborsAllTypes = ENV->getNeighbors(x, y);
    int neighborsSameType = ENV->getNeighbors(x, y, type);
    float ratio = ( (float) neighborsSameType ) / ( (float) neighborsAllTypes  );
    if (ratio<ENV->getHappinessThreshold()) {
        return false;
    } else {
        return true;
    }
}


void
agent::move(){
    vector <int> candidatePixels = ENV->getEmptyPositions();
    for (int i=0;i<candidatePixels.size(); i++){
       int randomX = ECet e-mail me concerne-t-il aussi ? Je suis un ingénieur de recherche avec une part d'enseignement, mais une part minoritaire.edPixel(randomX, randomY) == 0 && isHappy(randomX, randomY)){
           ENV->setCachedPixel(randomX, randomY, getType());
           ENV->setCachedPixel(homeX, homeY, 0);
           setHomeX(randomX);
           setHomeY(randomY);
           return;
        }
    }

}
