#ifndef AGENT_H
#define AGENT_H

#include <iostream>
#include <string>
#include <vector>

#include "env.h"

using namespace std;

class env;
class agent {
    public:
        agent();
        virtual ~agent();
        int getHomeX() { return homeX; }
        int getHomeY() { return homeY; }
        int getType() { return type; }
        string getTypeName() { return typeName; }
        env* getENV() { return ENV; }

        void setHomeX(int val) { homeX = val; }
        void setHomeY(int val) { homeY = val; }
        void setType(int val) { type = val; }
        void setTypeName(string val) { typeName = val; }
        void setENV(env* val) { ENV = val; }

        bool isHappy();
        bool isHappy(int x, int y);
        void move();



    protected:
    private:
        int homeX;
        int homeY;
        int type;
        string typeName;
        env* ENV;
};

#endif // AGENT_H
