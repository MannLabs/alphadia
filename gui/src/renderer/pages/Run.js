import * as React from 'react'

import { useMethod } from '../logic/context'

import { useTheme } from "@mui/material";
import { useRef, useEffect } from 'react'

import { FixedSizeList } from 'react-window';
import AutoSizer from 'react-virtualized-auto-sizer';
import InfiniteLoader from 'react-window-infinite-loader';

import { Box, TextField } from '@mui/material'
import { useProfile } from '../logic/profile';

function parseConsoleOutput(input, theme) {
    const escapeRegex = /\[(\d+;?\d*)m(.*?)\[(\d+)m/g;
    const matches = input.matchAll(escapeRegex);
  
    if (!matches || matches.length === 0) {
      return input; // No escape character pairs found, return original string
    }
    let result = [];
    let currentIndex = 0;
    for (const match of matches) {
      const startIndex = match.index;
      const text = match[2];
      const spanStyle = {color: getColorCode(match[1], theme), fontFamily: "Roboto Mono"}
      // Push the text before the escape character pair
      if (startIndex > currentIndex) {
        const plainText = input.substring(currentIndex, startIndex);
        result.push(plainText);
      }
      // Push the colored span with the text
      result.push(
        <span key={startIndex} style={spanStyle}>
          {text}
        </span>
      );
      currentIndex = startIndex + match[0].length;
    }
  
    // Push any remaining plain text after the last escape character pair
    if (currentIndex < input.length) {
      const remainingText = input.substring(currentIndex);
      result.push(remainingText);
    }
  
    return result;
  }

  function getColorCode(colorCode, theme) {
    // Map color codes to actual colors
    const colorMap = {
      '30;20': 'black',
      '31;20': theme.palette.mode === 'light' ? "rgb(200, 1, 0)" : 'rgb(250, 150, 136)' ,
      '32;20': theme.palette.mode === 'light' ?'rgb(76 211 26)' : 'rgb(168, 219, 114)',
      '33;20': theme.palette.mode === 'light' ?'rgb(253 167 0)' : 'rgb(254, 219, 119',
      '34;20': 'blue',
      '35;20': 'magenta',
      '36;20': 'cyan',
      '37;20': 'white',
      '38;20': 'inherit',
    };
  
    return colorMap[colorCode] || 'inherit'; // Default color is black
  }

  function applyCarriageReturns(items) {
    return items.reverse().reduce((acc, item) => {

        const lastItem = acc[acc.length - 1];
        if (lastItem && lastItem.endsWith('\r')) {
            acc[acc.length - 1] = lastItem.slice(0, -1)
        } else {
            acc.push(item);
        }
        return acc;
    }, []).reverse()
}          

const Output = () => {

    const [cmd, setCmd] = React.useState("")

    const method  = useMethod();
    const theme = useTheme();

    const [items, setItems] = React.useState([])
    const currentLengthRef = useRef(0);
    const [listRef, setListRef] = React.useState(null)
    const [scrollAttached, setScrollAttached] = React.useState(true)

    const profile = useProfile();

    useEffect(() => {
        currentLengthRef.current = items.length;
    })

    const updateItems = (currentLengthRef) => {
        window.electronAPI.getOutputRowsNew(-1,{limit:100, offset: currentLengthRef}).then((newItems) => {
            
            setItems( items => [...items, ...newItems]);
            //setItems((items)=>{applyCarriageReturns([...items, ...newItems])});
        });
    }
    
    React.useEffect(() => {
        setItems([]);
        currentLengthRef.current = 0;

        let isMounted = true;
        
        const interval = setInterval(() => {
            window.electronAPI.getOutputLengthNew(-1).then((length) => {
                if (isMounted){
                    if (length > currentLengthRef.current) {
                        updateItems(currentLengthRef.current);
                    }
                    if (length < currentLengthRef.current) {
                        setItems([]);
                        currentLengthRef.current = 0;
                    }
                }
            });
        }, 100);

        return () => {
            isMounted = false;
            clearInterval(interval);
        }


        
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
        const content = parseConsoleOutput((index >= items.length) ? "" : items[index], theme);
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