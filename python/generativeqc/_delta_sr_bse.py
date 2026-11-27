"""Content pins for the Basis Set Exchange records behind the ΔSR bases.

The scalar-relativistic correction (DLU-X2C-1e minus nonrelativistic) builds its
orbital and fitting bases from the installed ``basis_set_exchange`` package in
``_x2c1e.build_unc_x2c_basis_from_bse`` and ``_x2c1e.build_aux_basis_from_bse``.
Every record they may use is pinned here by its spherical element content
SHA-256 (:func:`generativeqc.basis.element_content_sha256`). A record with other
values, an export holding other elements, and a basis or element without a pin
are refused. The installed release is provenance only: any release whose
records keep these hashes is accepted.
"""

from __future__ import annotations

import json
from typing import Any

from .basis import element_content_sha256
from .basis_import import bse_element_records

VERIFIED_RELEASES = ("0.12",)

# Basis Set Exchange name -> atomic number -> element content SHA-256.
CONTENT_SHA256: dict[str, dict[int, str]] = {
    # Production orbital basis of the all-electron elements; the builder
    # decontracts it.
    "x2c-tzvpall": {
        1: "49d0b01688e5f5348cc1f3c36f58b233e90eb2b757c22c24ee7842ad247f3da4",
        5: "124a5bfe50e93adc71b11b1647aa300ea169de75be872cdaeea4c9135984be4c",
        6: "6aa5abd72ee6902c464f77249e5aa49f92950b91eb287aa2289340d0fca4ad32",
        7: "ce0199ce29d6216144b8dc986ca058d1ee5d0fad629e463909370359cbcc87c7",
        8: "17361446522ec4ebbfd49ce7f4af9840d476f6a3e8c6a768a356665e5ca205d4",
        9: "5f503ef13a53ed841dd0254b69cfc140104c91e47d0a236bc602960f615d2f03",
        11: "1140c64ae1997ac1c15d23341769aa3c5cb5c8ee00b6d8586071e85115fc973c",
        13: "3720d207270ca97561e9d42fb67043de33faa72717eb50d7f9da7a6c75838655",
        14: "dc7da49f7c48eba3c777782471974a3a154d75f58e5fa504f928973e658d92fb",
        15: "6e37782a2a6e1e15654a02b231dd05e96162cda3a8ab2d66f85b9153fd6c41a3",
        16: "99c3e62d503657b773f59f53c919f3baccd2060837afb1e8ea5cc6d164abb401",
        17: "dd69245f973302dd1455ea0addeb18429c8a74671aeba03eb905a2bb08f5b079",
    },
    # Production orbital basis and ECP of Br and I.
    "cc-pvdz-pp": {
        35: "637231493b296aa5018b5eb2bd49a586e6cfddbe00df5f0f73856b19437e0c92",
        53: "ff8ba8ef009ee2dd4ff1627474141a1db36e3fc434a130ac58356e32af561d18",
    },
    # Production fit of both the JK and the MP2 integrals.
    "def2-universal-jkfit": {
        1: "bf11c70bfe971463f3530ca810e50932a844bba80f0b54d1ea14cc6925fdef70",
        5: "202fdd1be1aadcdbdd608ebb8b357e22860075a976d8965a775fdbb3e97e7ffd",
        6: "8237aff18f432b9b36bd5f5cac3b5c9edbeee500441772e9eeed89ea283d3e4e",
        7: "68c9c7e18a7f34060bd692863b4bed6022aa66b03e1a755fb0dcffe65728136c",
        8: "a44f330642b01247ee9ae6e134564ef0de46dd49d0a959abc71d1888fb64dc9c",
        9: "93a0cf7e432fd8b5d005b9d47973e53d7938ada58d385e81e3adbd4191410cd8",
        11: "3208dd6501c72eac60860ec57cd43059c8002cc6c0f8c63694975b3c45fac1ac",
        13: "bd1830b0d21dee3994a8aa8e11581bea91013e5ab18d418b6252c422f32e30e1",
        14: "635422c889a9c1999bce00bd7ed4edd660327bc929b95809375a2c89db6bdd68",
        15: "2cbc5407ff9d6061702bbefc0d25410cfe3677ddc2c721ac618f37f73f8da540",
        16: "0c8eb7cd5c300fd88e9d679a3e1d8c7ab96e01bc5ce1d156b5f92fbfc3e44b29",
        17: "451362d8b2567f16fbc7ce1fd5f61b9913c5748bb50588c60db42cfa74b2fa2d",
        35: "e9b8c7a2c33b45ed7e18a8864485a7d3be77d856188bd039e40acc6b69631118",
        53: "805d83e482a851b19c8e6ceef1cf19144aec9b0424942d069c985430dd09e46b",
    },
    # Orbital basis of the reference tests.
    "x2c-svpall": {
        1: "3d535beedfba246a819d9879d3c46c141b0c1b19601593ae40d509460c1eaf48",
        5: "f61d04e379948e028eb9fe333697aa0a9f98b3b5f3e41dfcf16a3011926dd388",
        6: "634327e4217e9d026ea715aaac2a34dc5ea34c5edc2b0e020218acf28f355dc9",
        7: "d434af915133fef1fa532acd79e5143bd2e375d2198e516b7cf7f30da3283802",
        8: "294274275254c8399f8d1fcc17f827eaa662961e5470bda5406850e7ce0a2835",
        9: "75dadefd410ba5818ad06909fdc94d85e78d5506ef4033866e0c2a3000132de8",
        11: "b031013b1b1031567002e987ba091a389684dca917b09e63d88233de29b01d68",
        13: "81032fecf069866414a9126cdf4f9c2eb1fab446a1138fe8915712565ac5ff98",
        14: "10764532d33ed40f2f3f6dbe923abf25a2e8e350fcac70df342ec5d151e0c5ab",
        15: "2e902131911b10489d6434adf869819761f905936cf11cfbadf635649541f921",
        16: "f606061243016ac37072bf9b55f2f1e01a2af3989f25b5e4aeceb9931b87e151",
        17: "8c4d17018b0fcdad97f8b4ded538d715a96363fa639d5e65e904d0880a17a61b",
    },
    # MP2 fit (role ``mp2_fit``) of the reference tests.
    "def2-tzvp-rifit": {
        1: "c68854cda35ea7765117098784ffe361cf768a224da6d0d30842a2b4a63c2ef3",
        5: "b29d1314a096c9a17ecfe4cfa75ed737272a174df80fcaf45481a618c503b491",
        6: "f2d971b5b97e9f79112875922bb82e3232304d4468112ddc3caa00370facf46f",
        7: "7138c9358aa0ca4ca1c348e3f3b956f0d7105d02de31e398ee3de69c3b3607f6",
        8: "6d6af7c0f95c5cc0b7a10551a125ad3510594337afc5058841be5447afc40131",
        9: "763ebe1a1eed5ec015ed4544010bda1e48458d9f731af95b0d63222b1066183f",
        11: "e3d2642c4408e4960e788a53027f22409f025d2a5d1bc49bec9e75f448d4ab72",
        13: "8e882fc854c27a27faa204c8b6b34a35e8e41625c10d09b7e10006a230e24eaf",
        14: "b7922897b2566c3cdd0631887ea152c7a0764d63f4b66514cde3fe3faaf8ba92",
        15: "fbc0bf7300979eb9c6e1c07319bdad1948221adefc5cb4dea2f914cadacc5dab",
        16: "329953225613a195f54aaf7a05e05ec0d1c87e46e32518d523081f2662831c91",
        17: "c30fa79ea415f0a3e2de6c9ee224acb8bc30ac682f4bd5effea2297d537a0440",
        35: "37196a79f4c2861b7c41affc5b62907484c4bdf95eb167015bf2b6fc984f6501",
        53: "7eec702a44f6b33cb8ed13cf24c9d2ef353ee08d00025eefa6da97d2c98d9fb9",
    },
}


def load_pinned_bse_element(
    bse: Any, name: str, atomic_number: int
) -> tuple[str, dict[str, Any]]:
    """Load a pinned BSE element and return its JSON text and verified record."""
    expected = CONTENT_SHA256.get(name.lower(), {}).get(atomic_number)
    if expected is None:
        raise ValueError(
            f"basis_set_exchange {name} Z={atomic_number} has no pinned content "
            "SHA-256 for the ΔSR basis"
        )
    text = bse.get_basis(name, elements=[atomic_number], fmt="json")
    data = json.loads(text)
    records = bse_element_records(data)
    if set(records) != {atomic_number}:
        raise ValueError(
            f"basis_set_exchange {name} export holds elements Z={sorted(records)}. "
            f"Expected only Z={atomic_number}."
        )
    found = element_content_sha256(records[atomic_number], "spherical")
    if found != expected:
        raise ValueError(
            f"basis_set_exchange {name} Z={atomic_number} has content SHA-256 "
            f"{found}. Expected {expected}. Use the records verified with "
            f"basis_set_exchange {', '.join(VERIFIED_RELEASES)}."
        )
    return text, data["elements"][str(atomic_number)]
