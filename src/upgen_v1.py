#!/usr/bin/env python3
from dataclasses import dataclass
from enum import Enum, auto
import json
import math
import pprint
import random
import typing

from psf import Cipher

PARAMETERS = None

def set_parameters(parameters):
    global PARAMETERS
    PARAMETERS = parameters

def sample_from_parameters(k):
    val_prob_pairs_of_k = list(map(lambda k: (k["value"], k["probability"]),
                                   PARAMETERS["probabilities"][k]))

    return choice_probs(val_prob_pairs_of_k)

def range_of_parameters(k):
    minval = PARAMETERS["ranges"][k]["min"]
    maxval = PARAMETERS["ranges"][k]["max"]
    return range(minval, maxval)


def cipher_of_string(value: str):
    if value == "AES256GCM":
        return Cipher.AES256GCM
    elif value == "CHACHA20POLY1305":
        return Cipher.CHACHA20POLY1305
    else:
        raise NotImplementedError

# choose security parameter
def choose_security_parameter():
    return sample_from_parameters("sec_param")

def choose_cipher(secparam: int) -> Cipher:
    if (secparam == 128):
        return Cipher.AES128GCM
    elif (secparam == 256):
        return cipher_of_string(sample_from_parameters("256_bit_cipher"))
    else:
        raise NotImplementedError


def choose_subprotocol_nrounds(handshake_pattern):
    # flatten handshake pattern to search for optimistic data
    flat_handshake_pattern = [field for msg in handshake_pattern.key_patterns for field in msg]
    # potentially choose subprotocol rounds only if no optimistic data
    if (KeyPattern.EPHEMERAL_WITH_OPTIONAL_DATA not in flat_handshake_pattern):
        return sample_from_parameters("subprotocol_nrounds")
    else:
        return 0


def choose_subprotocol_sizes(subprotocol_nrounds: int):
    retval = []

    for _ in range(subprotocol_nrounds):
        for _ in range(2):
            retval.append(random.choice(range_of_parameters("subprotocol_round_sizes")))

    return retval


# Choose given values according to given probabilities
def choice_probs(val_prob_pairs):
    if not math.isclose(sum(map(lambda x: x[1], val_prob_pairs)), 1.0):
        raise ValueError("Probabilities in choice_probs() must sum to 1.")
    r = random.random()
    cumprob = 0
    for val, prob in val_prob_pairs:
        cumprob += prob
        if r <= cumprob:
            return val


# choose parameters of type field
def choose_type_field():
    type_field: typing.Any = dict()
    # Is type field encrypted?
    type_field["encrypted"] = sample_from_parameters("type_field_encrypted")
    # Choose number of type numbers
    type_field["num_type_nums"] = random.choice(range_of_parameters("ntype_nums"))
    # Choose starting type number
    type_field["start_num"] = sample_from_parameters("type_num_start_val")
    # Choose order of type numbers
    type_field["normal_types_first"] = sample_from_parameters("normal_types_first")

    return type_field


# choose parameters of length field
def choose_length_field():
    length_field = dict()
    length_field["encrypted"] = False
    length_field["length"] = sample_from_parameters("length_field_nbytes")
    length_field["mac_covered"] = sample_from_parameters("length_field_covered_by_mac")

    return length_field


# choose parameters of payload (i.e. message) padding
def choose_payload_padding_field(length_field):
    payload_padding_field: typing.Any = dict()
    payload_padding_field["encrypted"] = True
    # Choose length
    if length_field["length"] == 2:
        payload_padding_field["length"] = 2
    else:
        payload_padding_field["length"] = sample_from_parameters("padding_length")

    return payload_padding_field


# choose parameters of version field
def choose_version_field():
    version_field = dict()
    version_field["length"] = sample_from_parameters("version_field_nbytes")
    version_field["in_handshake"] = sample_from_parameters("version_field_in_handshake")
    version_field["encrypted"] = sample_from_parameters("version_field_encrypted")
    version_field["major_val"] = sample_from_parameters("major_version")
    if version_field["length"] == 2:
        version_field["minor_val"] = sample_from_parameters("minor_version")

    return version_field


# choose parameters of randomness field
def choose_randomness_field(secparam):
    randomness_field = dict()
    randomness_field["encrypted"] = False
    randomness_field["exists"] = sample_from_parameters("has_nonce")

    return randomness_field


# choose parameters of randomness field
def choose_extra_field(type_field, version_field):
    extra_field: typing.Any = dict()
    extra_field["encrypted"] = True
    extra_field["handshake_length"] = 0
    extra_field["data_length"] = 0
    if type_field["encrypted"] or version_field["encrypted"]:
        extra_field["handshake_length"] =\
            sample_from_parameters("extra_field_nbytes_handshake")
        extra_field["data_length"] =\
            sample_from_parameters("extra_field_nbytes_data")

    return extra_field


def choose_reserved_field():
    reserved_field = dict()
    reserved_field["encrypted"] = True
    reserved_field["length"] = sample_from_parameters("reserved_field_length_nbytes")

    return reserved_field


def choose_field_order(fields):
    field_order: dict[str, typing.Any] = dict()
    field_order["overall_structure"] = sample_from_parameters("overall_structure")
    encrypted_fields = []
    unencrypted_fields = []
    for field, parameters in fields.items():
        if "encrypted" in parameters and parameters["encrypted"] is True:
            encrypted_fields.append(field)
        else:
            unencrypted_fields.append(field)
    # put unencrypted fields in a random order
    # EDIT(rwails): Always start with type, length, version if possible.
    random.shuffle(unencrypted_fields)

    prefix_fields = [
        u for u in unencrypted_fields if u in ["type", "length", "version"]
    ]
    suffix_fields = [u for u in unencrypted_fields if u not in prefix_fields]

    random.shuffle(prefix_fields)
    random.shuffle(suffix_fields)

    unencrypted_fields = prefix_fields + suffix_fields

    # put encrypted fields in a random order
    random.shuffle(encrypted_fields)

    field_order["unencrypted"] = unencrypted_fields
    field_order["encrypted"] = encrypted_fields

    return field_order


class KeyPattern(Enum):
    EPHEMERAL = auto()
    EPHEMERAL_WITH_OPTIONAL_DATA = auto()
    STATIC = auto()
    ENCRYPTED_STATIC = auto()

    def __repr__(self):
        if self == KeyPattern.EPHEMERAL:
            return "EPHEMERAL"
        elif self == KeyPattern.EPHEMERAL_WITH_OPTIONAL_DATA:
            return "EPHEMERAL_WITH_OPTIONAL_DATA"
        elif self == KeyPattern.STATIC:
            return "STATIC"
        elif self == KeyPattern.ENCRYPTED_STATIC:
            return "ENCRYPTED_STATIC"
        else:
            raise NotImplementedError()


@dataclass
class HandshakePattern:
    key_patterns: list[list[KeyPattern]]


def choose_handshake_pattern() -> HandshakePattern:
    pattern1 = [
        [
            KeyPattern.EPHEMERAL,
        ],
        [
            KeyPattern.EPHEMERAL_WITH_OPTIONAL_DATA,
        ],
    ]
    pattern2 = [
        [
            KeyPattern.EPHEMERAL_WITH_OPTIONAL_DATA,
        ],
        [
            KeyPattern.EPHEMERAL_WITH_OPTIONAL_DATA,
        ],
    ]
    pattern3 = [
        [
            KeyPattern.EPHEMERAL,
        ],
        [
            KeyPattern.EPHEMERAL,
        ],
    ]
    pattern4 = [
        [
            KeyPattern.EPHEMERAL,
        ],
        [KeyPattern.EPHEMERAL, KeyPattern.STATIC],
    ]
    pattern5 = [
        [KeyPattern.EPHEMERAL, KeyPattern.STATIC],
        [
            KeyPattern.EPHEMERAL,
        ],
    ]
    pattern6 = [
        [KeyPattern.EPHEMERAL, KeyPattern.STATIC],
        [KeyPattern.EPHEMERAL, KeyPattern.STATIC],
    ]

    pattern7 = [
        [
            KeyPattern.EPHEMERAL,
        ],
        [
            KeyPattern.EPHEMERAL,
        ],
        [
            KeyPattern.ENCRYPTED_STATIC,
        ],
    ]

    pattern8 = [
        [
            KeyPattern.EPHEMERAL,
        ],
        [KeyPattern.EPHEMERAL, KeyPattern.STATIC],
        [
            KeyPattern.ENCRYPTED_STATIC,
        ],
    ]

    choices = (
        pattern1,
        pattern2,
        pattern3,
        pattern4,
        pattern5,
        pattern6,
        pattern7,
        pattern8,
    )

    def pattern_of_string(value: str):
        if value == "pattern1":
            return pattern1
        elif value == "pattern2":
            return pattern2
        elif value == "pattern3":
            return pattern3
        elif value == "pattern4":
            return pattern4
        elif value == "pattern5":
            return pattern5
        elif value == "pattern6":
            return pattern6
        elif value == "pattern7":
            return pattern7
        elif value == "pattern8":
            return pattern8
        else:
            raise NotImplementedError

    choice: list[list[KeyPattern]] =\
        pattern_of_string(sample_from_parameters("key_pattern"))

    return HandshakePattern(key_patterns=choice)


@dataclass
class ProtocolSettings:
    secparam: int
    fields: dict[str, typing.Any]
    field_order: dict[str, typing.Any]
    handshake_pattern: HandshakePattern
    key_encoding: str
    cipher: Cipher
    greeting: bool
    subprotocol_nrounds: int
    subprotocol_sizes: list[int]
    send_encrypted_cert: bool
    encrypted_cert_nbytes: int
    separate_length_fields: bool
    reversed_greeting: bool

    def __str__(self):
        return (
            f"Sec param: {self.secparam}\n"
            + f"Fields:\n{pprint.pformat(self.fields)}\n"
            + f"Field order:\n{pprint.pformat(self.field_order)}\n"
            + f"Handshake pattern: {self.handshake_pattern}\n"
            + f"Key encoding: {self.key_encoding}\n"
            + f"Encryption algorithm: {self.cipher}\n"
            + f"Subprotocol nrounds: {self.subprotocol_nrounds}\n"
            + f"Subprotocol sizes: {self.subprotocol_sizes}\n"
            + f"Send encrypted cert: {self.send_encrypted_cert}\n"
            + f"Encrypted cert nbytes: {self.encrypted_cert_nbytes}\n"
            + f"Separate length fields: {self.separate_length_fields}\n"
            + f"Reversed greeting: {self.reversed_greeting}"
        )


def sample_protocol_settings() -> ProtocolSettings:
    secparam = choose_security_parameter()
    cipher = choose_cipher(secparam)

    fields = {}
    fields["type"] = choose_type_field()
    fields["length"] = choose_length_field()
    fields["padding_length"] = choose_payload_padding_field(fields["length"])
    fields["version"] = choose_version_field()
    fields["nonce"] = choose_randomness_field(secparam)
    fields["extra"] = choose_extra_field(fields["type"], fields["version"])
    fields["reserved"] = choose_reserved_field()

    handshake_pattern = choose_handshake_pattern()
    subprotocol_nrounds = choose_subprotocol_nrounds(handshake_pattern)

    reversed_greeting = False

    greeting = sample_from_parameters("has_greeting")

    if greeting:
        reversed_greeting = sample_from_parameters("greeting_reversed")

    return ProtocolSettings(
        secparam=secparam,
        fields=fields,
        field_order=choose_field_order(fields),
        handshake_pattern=handshake_pattern,
        key_encoding=random.choice(("RAW", "DER", "PEM")),
        cipher=cipher,
        greeting=greeting,
        subprotocol_nrounds=subprotocol_nrounds,
        subprotocol_sizes=choose_subprotocol_sizes(subprotocol_nrounds),
        send_encrypted_cert=sample_from_parameters("send_encrypted_cert"),
        encrypted_cert_nbytes=random.choice(range_of_parameters("encrypted_cert_nbytes")),
        separate_length_fields=sample_from_parameters("separate_length_field"),
        reversed_greeting=reversed_greeting,
    )
