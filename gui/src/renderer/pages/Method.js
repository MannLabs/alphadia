import * as React from 'react'
import { useMethod } from '../logic/context'

import { Masonry } from '@mui/lab';
import { ParameterGroup } from '../components'

const Method = () => {

    const method  = useMethod();

    return (
    <Masonry columns={{ xs: 1, sm: 2, md: 2, lg: 3, xl: 3 }} spacing={1}>
        {method.config.map((parameterGroup, index) => (
                <ParameterGroup
                    parameterGroup={parameterGroup}
                    index={index}
                />
        ))}
    </Masonry>
    )
}

export default Method
