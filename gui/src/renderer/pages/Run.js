import * as React from 'react'

import { useMethod } from '../logic/context'
import { useRef, useEffect } from 'react'
import { FixedSizeList } from 'react-window';
import AutoSizer from 'react-virtualized-auto-sizer';
import InfiniteLoader from 'react-window-infinite-loader';

import { Box, TextField } from '@mui/material'

const Output = () => {

    const [cmd, setCmd] = React.useState("")

    const method  = useMethod();

    const [items, setItems] = React.useState([])
    const currentLengthRef = useRef(0);
    const [listRef, setListRef] = React.useState(null)
    const [scrollAttached, setScrollAttached] = React.useState(true)

    useEffect(() => {
        currentLengthRef.current = items.length;
    })

    const updateItems = (currentLengthRef) => {
        window.electronAPI.getOutputRows({limit:100, offset: currentLengthRef}).then((newItems) => {
            setItems( items => [...items, ...newItems]);
        });
    }
    
    React.useEffect(() => {
        console.log(method);

        setInterval(() => {
            window.electronAPI.getOutputLength().then((length) => {
                if (length > currentLengthRef.current) {
                    updateItems(currentLengthRef.current);
                }
            });
        }, 100);
        
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);


    const onScroll = ({ scrollDirection, scrollOffset, scrollUpdateWasRequested }) => {
        // scrolling is attached to the end of the list
        // continue to keep it attached
        if (scrollUpdateWasRequested) {
            setScrollAttached(true);

        // scrolling was requested by the user
        } else {

            // The user is scrolling forward and the end of the list is near
            // attach scrolling to the end of the list
            if (scrollDirection === "forward") {
                
                // scrollOffset is the number of pixels from the top of the list
                // listRef?.props?.height is the height of the list in pixels
                // listRef?.props?.itemSize is the height of each item in pixels
                // bottomElement is the index of the bottom element in the list
                const bottomElement = (scrollOffset + listRef?.props?.height) / listRef?.props?.itemSize;
                
                // if the bottom element is within 5 elements of the end of the list
                if ((items.length - bottomElement) < 3)
                    setScrollAttached(true);

            // The user is scrolling somewhere in the middle of the list
            // detach scrolling from the end of the list
            } else {
                setScrollAttached(false);
            }
        }
    }

    // scroll to the end of the list when new items are added
    useEffect(() => {
        if (scrollAttached) {
            listRef?.scrollToItem?.(items.length, 'end')
        }
        
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [listRef, items.length])

    const Row = ({ index, style }) => {
        const content = (index >= items.length) ? "" : items[index];
        return (
            <Box style={style} sx={{fontSize: "inherit", fontFamily: "Roboto Mono", lineHeight: "1.0rem", textWrap: "nowrap", overflow: "hidden"}}>
                {content}
            </Box>
        )
    };

    return (
    <>
    <TextField
        id="outlined-number"
        type="text"
        variant="standard"
        size="small"
        sx = {{width: "100%"}}
        value={cmd}
        onChange={(event) => {setCmd(event.target.value)}}
        onKeyDown={(event) => {
            if (event.key === 'Enter'){
                window.electronAPI.runCommand(cmd).catch((error) => {
                    console.log(error);
                });
                setCmd("");
         }}}
    />
    <Box sx={{
        height: "calc(100% - 90px)",
        whiteSpace: "break-spaces",
        overflow: "hiodden",
        fontFamily: "Roboto Mono",
        fontSize: "0.7rem",
        lineHeight: "1.0rem",
        }}>

        <InfiniteLoader
            isItemLoaded={(index) => (index < items.length)}
            loadMoreItems={()=>{}}
            itemCount={items.length}
            >
            {({ onItemsRendered, ref: setRef }) => (
        <AutoSizer>
            {({ height, width }) => (
                <FixedSizeList
                    className="List"
                    height={height}
                    itemCount={items.length}
                    itemSize={15}
                    width={width}
                    onItemsRendered={onItemsRendered}
                    onScroll={onScroll}
                    ref={(list) => {

                    setRef?.(list);
                    setListRef(list);
                    }}
                >
                    {Row}
                </FixedSizeList>
            )}
        </AutoSizer>
            )}
        </InfiniteLoader>
    </Box>
    </>
    )}

export default Output