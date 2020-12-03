class EnvPathException(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message


class EmptyStringException(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message


class InavlidFormatException(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message


class InavlidYamlFileException(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message


class InavlidJSONFileException(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message


class InvalidFeatureException(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message


class FileError(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message


class FeatureExtractionException(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message


class POSTagExtractionFailedException(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message


class DependencyParsingTreeException(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message


class ConstituencyTreeParsingException(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message


class YngveTreeConstructionException(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message


class LexicoSemanticParsingException(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message


class MorphoSyntacticParsingException(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message


class SyntacticParsingException(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message


class DiscourseParsingException(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message
