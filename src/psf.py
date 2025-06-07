#!/usr/bin/env python3

from dataclasses import dataclass
from enum import Enum, auto
import typing
import util


def auto_str_repr(cls):
    if not hasattr(cls, "to_str"):
        raise AttributeError(f"{cls.__name__} does not have a 'to_str' method")

    cls.__str__ = cls.to_str
    cls.__repr__ = cls.to_str
    return cls


@auto_str_repr
class NumericType(Enum):
    U8 = auto()
    U16 = auto()
    U32 = auto()
    U64 = auto()
    I8 = auto()
    I16 = auto()
    I32 = auto()
    I64 = auto()

    def to_str(self) -> str:
        return {
            NumericType.U8: "u8",
            NumericType.U16: "u16",
            NumericType.U32: "u32",
            NumericType.U64: "u64",
            NumericType.I8: "i8",
            NumericType.I16: "i16",
            NumericType.I32: "i32",
            NumericType.I64: "i64",
        }[self]

    def repr_nbytes(self) -> int:
        return {
            NumericType.U8: 1,
            NumericType.U16: 2,
            NumericType.U32: 4,
            NumericType.U64: 8,
            NumericType.I8: 1,
            NumericType.I16: 2,
            NumericType.I32: 4,
            NumericType.I64: 8,
        }[self]


@auto_str_repr
class Bool:
    def to_str(self) -> str:
        return "bool"


@auto_str_repr
class Char:
    def to_str(self) -> str:
        return "char"


PrimitiveType = typing.Union[NumericType | Bool | Char]

Identifier = typing.NewType("Identifier", str)

HexLiteral = typing.NewType("HexLiteral", str)


@auto_str_repr
@dataclass
class PrimitiveArray:
    primitive_type: PrimitiveType
    nelems: int

    def to_str(self) -> str:
        return "[ " + str(self.primitive_type) + " ; " + str(self.nelems) + " ]"

    def repr_nbytes(self) -> int:
        return self.primitive_type.repr_nbytes() * self.nelems


@auto_str_repr
@dataclass
class DynamicArray:
    id: Identifier

    def to_str(self) -> str:
        return "[ " + str(NumericType.U8) + " ; " + str(self.id) + ".size_of" + " ]"


Array = typing.Union[PrimitiveArray | DynamicArray]
TypeValue = typing.Union[PrimitiveType | Array]


@auto_str_repr
@dataclass
class Field:
    name: Identifier
    type_value: TypeValue

    def to_str(self) -> str:
        return (
            "{ "
            + "NAME:\t"
            + str(self.name)
            + " ; "
            + "TYPE:\t"
            + str(self.type_value)
            + " }"
        )


@auto_str_repr
@dataclass
class Format:
    name: Identifier
    fields: list[Field]

    def to_str(self) -> str:
        return (
            "DEFINE " + str(self.name) + "\n" + ",\n".join(map(str, self.fields)) + ";"
        )


@auto_str_repr
@dataclass
class FixedStringSemantic:
    value: str

    def to_str(self) -> str:
        return f'FIXED_STRING ("{self.value}")'


@auto_str_repr
@dataclass
class FixedBytesSemantic:
    value: HexLiteral

    def to_str(self) -> str:
        return "FIXED_BYTES (" + self.value + ")"


@auto_str_repr
@dataclass
class RandomSemantic:
    value: int

    def to_str(self) -> str:
        return "RANDOM (" + str(self.value) + ")"


@auto_str_repr
class PubkeyEncoding(Enum):
    RAW = auto()
    DER = auto()
    PEM = auto()

    def to_str(self) -> str:
        match self:
            case PubkeyEncoding.RAW:
                return "RAW"
            case PubkeyEncoding.DER:
                return "DER"
            case PubkeyEncoding.PEM:
                return "PEM"
            case _:
                raise NotImplementedError()

    @staticmethod
    def of_str(s: str) -> "PubkeyEncoding":
        match s:
            case "RAW":
                return PubkeyEncoding.RAW
            case "DER":
                return PubkeyEncoding.DER
            case "PEM":
                return PubkeyEncoding.PEM
            case _:
                raise ValueError()

    def encoding_size_nbytes(self) -> int:
        match self:
            case PubkeyEncoding.RAW:
                return 32
            case PubkeyEncoding.DER:
                return 44
            case PubkeyEncoding.PEM:
                return 115
            case _:
                raise NotImplementedError()


@auto_str_repr
@dataclass
class PubkeySemantic:
    value: PubkeyEncoding

    def to_str(self) -> str:
        return "PUBKEY (" + str(self.value.to_str()) + ")"


@auto_str_repr
class SemanticValue(Enum):
    PADDING = auto()
    PAYLOAD = auto()
    LENGTH = auto()
    PADDING_LENGTH = auto()

    def to_str(self) -> str:
        return {
            SemanticValue.PADDING: "PADDING",
            SemanticValue.PAYLOAD: "PAYLOAD",
            SemanticValue.LENGTH: "LENGTH",
            SemanticValue.PADDING_LENGTH: "PADDING_LENGTH",
        }[self]


FieldSemantic = typing.Union[
    FixedStringSemantic
    | FixedBytesSemantic
    | RandomSemantic
    | PubkeySemantic
    | SemanticValue
]


@auto_str_repr
@dataclass
class SemanticBinding:
    format_id: Identifier
    field_id: Identifier
    semantic: FieldSemantic

    def to_str(self) -> str:
        return (
            "{ "
            + "FORMAT:\t"
            + str(self.format_id)
            + " ; "
            + "FIELD:\t"
            + str(self.field_id)
            + " ; "
            + "SEMANTIC:\t"
            + str(self.semantic)
            + " } ;"
        )


@auto_str_repr
class Role(Enum):
    CLIENT = auto()
    SERVER = auto()

    def to_str(self) -> str:
        return {Role.CLIENT: "CLIENT", Role.SERVER: "SERVER"}[self]

    @staticmethod
    def of_str(value: str):
        return {"client": Role.CLIENT, "server": Role.SERVER}[value]


@auto_str_repr
class Phase(Enum):
    HANDSHAKE = auto()
    DATA = auto()

    def to_str(self) -> str:
        return {Phase.HANDSHAKE: "HANDSHAKE", Phase.DATA: "DATA"}[self]


@auto_str_repr
@dataclass
class SequenceSpecifier:
    role: Role
    phase: Phase
    format_id: Identifier

    def to_str(self) -> str:
        return f"{{ ROLE:\t{self.role} ; PHASE:\t{self.phase} ; FORMAT:\t{self.format_id} }} ;"


@auto_str_repr
@dataclass
class PasswordAssignment:
    value: str

    def to_str(self) -> str:
        return f'PASSWORD = "{self.value}" ;'


@auto_str_repr
class Cipher(Enum):
    CHACHA20POLY1305 = auto()
    AES128GCM = auto()
    AES256GCM = auto()

    def to_str(self) -> str:
        match self:
            case Cipher.CHACHA20POLY1305:
                return "CHACHA20-POLY1305"
            case Cipher.AES128GCM:
                return "AES128GCM"
            case Cipher.AES256GCM:
                return "AES256GCM"

    def block_size_nbytes(self) -> typing.Optional[int]:
        match self:
            case Cipher.CHACHA20POLY1305:
                return None
            case Cipher.AES128GCM | Cipher.AES256GCM:
                return 16

    def key_size_nbytes(self) -> int:
        match self:
            case Cipher.AES128GCM:
                return 16
            case Cipher.AES256GCM | Cipher.CHACHA20POLY1305:
                return 32

    def iv_length_nbytes(self) -> int:
        match self:
            case Cipher.CHACHA20POLY1305 | Cipher.AES128GCM | Cipher.AES256GCM:
                return 12

    def mac_tag_nbytes(self) -> int:
        match self:
            case Cipher.CHACHA20POLY1305 | Cipher.AES128GCM | Cipher.AES256GCM:
                return 16


@auto_str_repr
@dataclass
class CipherAssignment:
    cipher: Cipher

    def to_str(self) -> str:
        return f"CIPHER = {self.cipher} ;"


@auto_str_repr
@dataclass
class EncryptionFormatBinding:
    encrypt_id: Identifier
    from_id: Identifier

    def to_str(self) -> str:
        return f"ENCRYPT {self.encrypt_id} FROM {self.from_id}"


@auto_str_repr
@dataclass
class EncryptionFieldDirective:
    ptext_id: Identifier
    ctext_id: Identifier
    mac_id: typing.Optional[Identifier]

    def to_str(self) -> str:
        mac_id = "NULL"

        if self.mac_id is not None:
            mac_id = self.mac_id

        return (
            f"{{ PTEXT : {self.ptext_id} ; CTEXT : {self.ctext_id} ; MAC : {mac_id} }}"
        )


@auto_str_repr
@dataclass
class EncryptionDirectives:
    enc_fmt_bnd: EncryptionFormatBinding
    enc_fmt_dirs: list[EncryptionFieldDirective]

    def to_str(self) -> str:
        return (
            str(self.enc_fmt_bnd)
            + "\n"
            + ",\n".join(map(str, self.enc_fmt_dirs))
            + " ;"
        )


@auto_str_repr
@dataclass
class CryptoSegment:
    pass_assn: typing.Optional[PasswordAssignment]
    cipher_assn: CipherAssignment
    enc_dirs: list[EncryptionDirectives]

    def to_str(self) -> str:
        return (
            "\n@SEGMENT.CRYPTO\n"
            + util.str_of_optional(self.pass_assn)
            + "\n"
            + str(self.cipher_assn)
            + "\n"
            + "\n".join(map(str, self.enc_dirs))
        )

@auto_str_repr
@dataclass
class SeparateLengthFieldSetting:
    separate_length_field: Bool

    def to_str(self) -> str:

        value = "true" if self.separate_length_field else "false"

        return f"SEPARATE_LENGTH_FIELD = {value} ;"

@auto_str_repr
@dataclass
class OptionsSegment:
    separate_length_field: typing.Optional[SeparateLengthFieldSetting]

    def to_str(self) -> str:
        return (
            "\n@SEGMENT.OPTIONS\n"
            + util.str_of_optional(self.separate_length_field)
        )

@auto_str_repr
@dataclass
class ProtocolSpecificationFile:
    formats: list[Format]
    semantics: list[SemanticBinding]
    sequence: list[SequenceSpecifier]
    crypto: typing.Optional[CryptoSegment]
    options: typing.Optional[OptionsSegment]

    def to_str(self) -> str:
        return (
            "@SEGMENT.FORMATS\n"
            + "\n".join(map(str, self.formats))
            + "\n@SEGMENT.SEMANTICS\n"
            + "\n".join(map(str, self.semantics))
            + "\n@SEGMENT.SEQUENCE\n"
            + "\n".join(map(str, self.sequence))
            + util.str_of_optional(self.crypto)
            + util.str_of_optional(self.options)
        )
