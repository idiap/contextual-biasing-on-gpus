#include <iostream>
#include <vector>

typedef unsigned int uint32;

struct _BFSState {
        _BFSState(uint32 state, uint32 prev_state, uint32 output, uint32 token_idx, bool is_first=false) {
            is_first_ = is_first;
            state_ = state;
            prev_state_ = prev_state;
            output_ = output;
            token_idx_ = token_idx;
        }

        uint32 state_;
        uint32 prev_state_;
        bool is_first_ = true;
        uint32 output_;
        uint32 token_idx_;
};
typedef struct _BFSState BFSState;
typedef std::vector<BFSState> Path;

int main() {
    BFSState s(0, 0, 0, 0);
    return 0;
}
