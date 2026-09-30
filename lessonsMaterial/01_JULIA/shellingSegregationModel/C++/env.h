#ifndef ENV_H
#define ENV_H

#include <iostream>

#include <string>
#include <sstream>
#include <vector>

#include "agent.h"


using namespace std;

class agent;

class env {
    public:
        env();
        virtual ~env();
        void init();
        void getParameters();
        void createEnvironment();
        void randomlyLocateAgents();

        void step(int stepn);
        void print(int stepn);

        int posToX(int pos){return pos%cols;};
        int posToY(int pos){return pos/cols;};
        int xyToPos(int x, int y){return x+y*cols;};

        int askInteger (int maxvalue=30, int defaultValue=0); // 30 è l'argomento di default e 0 il valore restituito di default
        float askFloat (float maxvalue=1.0, float defaultValue=0.5);
        string askString(string defaultValue="");

        int getNeighbors(int x, int y);
        int getNeighbors(int x, int y, int type);

        int getSteps(){return totalSteps;}
        int getWatchRings(){return watchRings;}
        int getRows(){return rows;}
        int getCols(){return cols;}
        int getWaitSeconds(){return waitSeconds;}
        float getHappinessThreshold(){return happinessThreshold;}
        agent* getAgent(int x, int y) {return agents[xyToPos(x,y)];};
        int getCachedPixel(int x, int y){return cachedPixels[xyToPos(x,y)]; };
        void setCachedPixel (int x, int y, int type){cachedPixels[xyToPos(x,y)] = type;};
        void refreshCachedPixels();
        vector <int> getEmptyPositions();


        int                 s2i (string string_h)          const; ///<  string  to integer conversion
        float               s2f (string string_h)          const; ///<  string  to float   conversion
        bool                s2b (string string_h)          const; ///<  string  to bool    conversion
        string              i2s (int int_h)                const; ///<  integer to string  conversion
        string              d2s (double double_h)          const; ///<  double  to string  conversion
        string              b2s (bool bool_h)              const; ///<  bool    to string  conversion


    protected:
    private:
        vector <int> cachedPixels;
        vector <agent *> agents;
        int rows;
        int cols;
        int nAg1;
        int nAg2;
        string nameAg1;
        string nameAg2;
        float happinessThreshold;
        int watchRings;
        int totalSteps;
        int displaySteps;
        int waitSeconds;

};

#endif // ENV_H
