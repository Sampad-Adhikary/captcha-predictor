class CharMapping:
    vocabulary = ['0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
    char_to_num = {c:i for i,c in enumerate(vocabulary)}
    # num_to_char = {0:'0',1:'1',}
    num_to_char = {str(v): k for k, v in char_to_num.items()}

    @classmethod
    def get_mapping(cls):
        return cls.char_to_num
    
    @classmethod
    def get_num_to_char(cls):
        return cls.num_to_char

    @classmethod
    def encode(cls, text):
        """Convert a string into a list of numbers based on the mapping."""
        return [cls.char_to_num[char] for char in text if char in cls.char_to_num]

    @classmethod
    def decode(cls, numbers):
        """Convert a list of numbers back into a string."""
        num_to_char = {v: k for k, v in cls.char_to_num.items()}
        return "".join(num_to_char[num] for num in numbers if num in num_to_char)