import numpy as np

class AI:
    def __init__(self, rules=None, input_size=6, num_states=32):
        self.input_size = input_size
        self.num_states = num_states
        self.head = 0
        # Corrected bits_per_rule calculation
        self.bits_per_rule = (self.num_states - 1).bit_length() + 1  # Bits for next_state + write_symbol
        self.rules = rules if rules is not None else self.init_binary_rules()

    def reset(self):
        self.head = 0

    def init_binary_rules(self):
        num_rules = self.num_states * 2
        rules = np.zeros((num_rules, self.bits_per_rule), dtype=int)
        for state in range(self.num_states):
            for symbol in range(2):
                rule_index = state * 2 + symbol
                
                # Generate a valid next_state within [0, num_states - 1]
                next_state = np.random.randint(0, self.num_states)
                write_symbol = np.random.choice([0, 1], replace=False)
                
                # Convert next_state to binary bits
                next_state_bits = list(map(int, bin(next_state)[2:].zfill(self.bits_per_rule - 1)))
                
                # Write to the rule
                rules[rule_index, :-1] = next_state_bits  # Store state bits
                rules[rule_index, -1] = write_symbol     # Store write symbol
        return rules

    def lookup_rule(self, state, symbol):
        # Ensure valid state and symbol
        if not (0 <= state < self.num_states):
            raise ValueError(f"Invalid state {state}, must be in range [0, {self.num_states - 1}]")
        if not (0 <= symbol < 2):
            raise ValueError(f"Invalid symbol {symbol}, must be 0 or 1")
        
        # Locate the rule chunk
        rule_index = state * 2 + symbol  # Sequential mapping
        rule_bits = self.rules[rule_index]
        
        # Decode `next_state` from the first `self.bits_per_rule - 1` bits
        next_state_bits = rule_bits[:-1]
        next_state = next_state_bits.dot(1 << np.arange(next_state_bits.size)[::-1])
        
        # Decode `write_symbol` (last bit)
        write_symbol = rule_bits[-1]
        return int(next_state), write_symbol

    def run(self, input_data):
        output = np.zeros(input_data.shape, dtype=int)
        for i in range(self.input_size):
            symbol = input_data[i]
            # Fetch next state and write symbol using the helper
            next_state, write_symbol = self.lookup_rule(self.head, symbol)
            self.head = next_state
            output[i] = write_symbol
        
        left = output[:output.size // 2].sum()
        right = output[output.size // 2:].sum()
        if left == right:
            return 0
        elif left > right:
            return 1
        else:
            return -1
    
    def mutate_rules(self, mutation_rate=0.05):
        mutation_mask = np.random.rand(*self.rules.shape) < mutation_rate
        mutated_rules = self.rules.copy()
        mutated_rules[mutation_mask] = np.random.randint(0, 2, size=mutation_mask.sum())
        self.rules = mutated_rules
