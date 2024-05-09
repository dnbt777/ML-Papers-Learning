import random


def generate_rules_string(charset, length=40):
    result = []
    i = 0
    while i < length:
        if i >= 2 and result[-2:] == ['a', 'a']:
            # Ensure that four letters later, there are three 'b's
            if i + 4 < length:
                # Fill the next character randomly but not 'a'
                next_char = random.choice([ch for ch in charset if ch != 'a'])
                result.append(next_char)
                i += 1
                if i < length:
                    # Add three 'b's four letters after the double 'a'
                    if i + 3 == length - 4:
                        result.extend(['b', 'b', 'b'])
                        i += 3
                    else:
                        # Fill with random characters until the position to insert 'b's
                        while i < length - 4:
                            next_char = random.choice([ch for ch in charset if ch != 'd' or (result[-1] != 'c')])
                            result.append(next_char)
                            i += 1
                        result.extend(['b', 'b', 'b'])
                        i += 3
            else:
                # Not enough space to enforce the rule, fill the rest with valid random characters
                while i < length:
                    next_char = random.choice([ch for ch in charset if ch != 'd' or (result[-1] != 'c')])
                    result.append(next_char)
                    i += 1
        else:
            # Choose the next character
            if result and result[-1] == 'c':
                # 'c' cannot be followed by 'd'
                next_char = random.choice([ch for ch in charset if ch != 'd'])
            else:
                next_char = random.choice(charset)
            
            # Handle 'b' not being in even groups
            if next_char == 'b':
                if result and result[-1] == 'b':
                    # If the last character was 'b', add one more 'b' to make it an odd group
                    result.append('b')
                    i += 1
                    if i < length:
                        # Add a different character to break the sequence
                        next_char = random.choice([ch for ch in charset if ch != 'b'])
                else:
                    # Start a new group of 'b's, add just one
                    result.append('b')
                    i += 1
                    continue
            result.append(next_char)
            i += 1

    return ''.join(result)

